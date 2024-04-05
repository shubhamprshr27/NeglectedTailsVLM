import sqlite3
import os
import pyarrow.parquet as pq
import multiprocessing
from multiprocessing import Pool, set_start_method
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import argparse
import json
import pickle
import re
import gc
import sys

"""
1. Normalize text.
2. Find alternate names for labels with single names in imagenet.
3. Next, normalized search for both human engineered, and original Imagenet-1K labels.
"""
NUM_TABLES = {
    'LAION400M.db': 32,
    'LAION2B.db': 128
}


class LaionParser():
    def __init__(self, database, mode='single', 
                 source= './', max_threads:int = 16, 
                 num_proc: int = 32, prefix: str = None,
                 matching_strategy: str = 'RELAXED') -> None:
        self.database_name = database
        self.database_path = f'{database}.db'
        self.mode = mode
        self.source = source
        if self.mode == 'single':
            self.conn = self.__connect__()
        self.start_time = time.time()
        self.num_tables = NUM_TABLES[self.database_path]
        self.max_theads = max_threads
        self.num_proc = min(num_proc, multiprocessing.cpu_count())
        self.prefix = prefix
        self.matching_strategy = matching_strategy

        if not os.path.exists(os.path.join(self.source,self.database_path)):
            print(f'SQLite representation of {self.database_name} not found. Making now.')
            for shard in range(self.num_tables):
                df = self.create_table(shard)
                self.create_fts_table(df=df, shard=shard)
            print('Setup complete.')
    
    # Needed only once. Don't execute later.
    def create_table(self, shard):
        shard_id = str(shard).zfill(5)
        try:
            parquet_file = pq.ParquetFile(f'./{self.database_name.lower()}/part-{shard_id}-5114fd87-297e-42b0-9d11-50f1df323dfa-c000.snappy.parquet')
        except:
            sys.exit('Error in finding LAION metadata files, ensure the path above is correct')
        df = parquet_file.read().to_pandas()
        df.to_sql(f'part{shard}', self.conn)
        self.conn.commit()
        return df
    
    def find_by_id(self, rowid: str, shard: int, column: str):
        cursor = self.conn.cursor()
        cursor.execute(f'''
            select {column}, nsfw from part{shard} where rowid = {rowid};
        ''')
        result = cursor.fetchone()
        return result
    
    # Needed only once. Don't execute later.
    def create_fts_table(self, df, shard):
        print(f'Creating FTS table - {shard}')
        cursor = self.conn.cursor()
        texts = df['TEXT'].tolist()
        # Normalize text and save.
        texts_norm = [text.replace('"', '').replace("'", '').replace("-", ' ').replace('_',' ') if text else "N.A." for text in texts]
        texts_norm = [tuple([text],) for text in texts_norm]
        cursor.execute(f'''
            CREATE VIRTUAL TABLE _fts{shard} USING FTS5(TEXT);
        ''') 
        self.conn.commit() 
        cursor.executemany(f"INSERT INTO _fts{shard} (TEXT) values(?)", texts_norm)
        self.conn.commit()
    
    def get_label_stats(self, label, shard, cursor):
        label = label
        parsed_label = self.parse_multi_words(label)
        if ("(" in label and ")" in label) or "." in label or '/' in label:
            cursor.execute(f'''
                select rowid, text from _fts{shard} where _fts{shard} MATCH '"{parsed_label}"' ORDER BY RANK;
            ''')
        else:
            cursor.execute(f'''
                select rowid, text from _fts{shard} where _fts{shard} MATCH '{parsed_label}' ORDER BY RANK;
            ''')
        matches = cursor.fetchall()
        return matches
    
    def parse_multi_words(self, text: str):
        text = clean_text(text=text)
        text = text.replace(" ", " + ")
        if "(" in text and ")" in text:
            text = text.replace("(", "").replace(")", "")
        if self.prefix is not None:
            text = self.prefix +" "+ text
        return text

    def __connect__(self):
        return sqlite3.connect(os.path.join(self.source, self.database_path))

    def __get_frequency_worker__(self, args):
        key, metrics = args
        names = list(metrics['alternates'].keys())
        conn = self.__connect__()
        cursor = conn.cursor()
        all_matches = 0
        for name in names:
            total_matches = 0
            for i in range(self.num_tables):
                total_matches += self.get_label_stats(name, shard=i, cursor=cursor)
            all_matches += total_matches
            metrics['alternates'][name] = total_matches
        conn.close()
        print(f'Frequency counted for: {key} -  time: {time.time()-self.start_time} - matches: {all_matches}')
        return {key: metrics}
        
    def __get_text_worker__(self, args):
        conn = self.__connect__()
        cursor = conn.cursor()
        key, metrics = args
        sorted_metrics = sorted(metrics['alternates'].items(), key=lambda x: x[1])
        name_stack = [item[0] for item in sorted_metrics]
        searched = set()
        total_matches = set()
        caption_set = set()
        while len(name_stack) != 0:
            name = name_stack.pop()
            og_name = "".join(name)
            name = clean_text(name)
            searched.add(name)
            name_count = 0
            name = self.parse_multi_words(name)
            try:
                for shard in range(self.num_tables):
                    if ("(" in name and ")" in name) or "." in name or '/' in name:
                        cursor.execute(f'''
                            select rowid, text from _fts{shard} where _fts{shard} MATCH '"{name}"' ORDER BY RANK;
                        ''')
                    else:
                        cursor.execute(f'''
                            select rowid, text from _fts{shard} where _fts{shard} MATCH '{name}' ORDER BY RANK;
                        ''')
                    matches = cursor.fetchall()
                    new_matches = [(shard,)+ match for match in matches]
                    name_count += len(new_matches)
                    for match_i in new_matches:
                        if match_i not in caption_set:
                            caption_set.add(match_i)
                            total_matches.add((og_name,)+match_i)
                metrics['alternates'][og_name] = name_count
            except:
                print('exception', og_name, key)
        conn.close()
        return ({key: metrics}, {key: total_matches})

    def get_freq_parallel(self, metrics):
        metrics_flattened = []
        for item in metrics.items():
            key, val = item
            metrics_flattened.append((key,val))
        set_start_method('fork')
        with ThreadPoolExecutor(self.max_theads) as pool:
            futures_to_results = {
                pool.submit(self.__get_frequency_worker__,(key, val)): (key,val) for (key,val) in metrics_flattened
            }
            results = pool.map(self.__get_frequency_worker__, metrics_flattened)

        for result in results:
            metrics.update(result)
        print(f'Total time: {time.time()-self.start_time}')
        return metrics

    def get_text_parallel(self, metrics):
        metrics_flattened = []
        for item in metrics.items():
            key, val = item
            metrics_flattened.append((key,val))

        retrieved_captions = {}

        infrequent_classes = []
        processed_classes = 0
        with ThreadPoolExecutor(self.max_theads) as pool:
            futures_to_metrics = {
                pool.submit(self.__get_text_worker__,(key, val)): (key,val) for (key,val) in metrics_flattened
            }

            for future in as_completed(futures_to_metrics):
                result = future.result()
                (key, value) = futures_to_metrics.pop(future)
                updated_metrics, matches = result
                metrics.update(updated_metrics)
                retrieved_captions.update(matches)
                metrics[key]['most_common_name'] = find_most_common_name(metrics=metrics[key], matching_strategy=self.matching_strategy)
                metrics[key]['actual_freq'] = len(retrieved_captions[key])
                if metrics[key]['actual_freq'] < 100:
                    infrequent_classes.append(f"name: {metrics[key]['name']} class: {key} freq: {len(retrieved_captions[key])}")
                processed_classes +=1
                print(f'Total processed: {processed_classes} - Processed {key} -> label: {value["name"]}, freq: {metrics[key]["actual_freq"]} - time: {time.time() - self.start_time}')
                gc.collect()
        
        return retrieved_captions, metrics, infrequent_classes

def clean_text(text: str):
    return text.strip().replace("'",'').replace('"','').replace('-', ' ').replace('_', ' ').replace("  ", ' ').lower()

def find_most_common_name(metrics:dict, matching_strategy:str = 'RELAXED'):
    # maintain a copy
    official_name_og = "".join(official_name)

    # Order from the official name.
    alternates_ordered = dict(sorted(metrics['alternates'].items(), key=lambda x: x[1], reverse=True))
    most_common_name = "".join(official_name)
    if official_name not in alternates_ordered:
        most_common_name_freq = alternates_ordered[clean_text(official_name)]
    else:
        most_common_name_freq = alternates_ordered[official_name]
    official_name = clean_text(official_name)
    official_name = re.sub(r'[^\w\s]', '', official_name)
    official_name_split = set(official_name.split())
    
    for alternate_name in alternates_ordered.keys():
        alternate_name_freq = alternates_ordered[alternate_name]
        alternate_name_og = "".join(alternate_name)
        alternate_name = clean_text(alternate_name)
        alternate_name = re.sub(r'[^\w\s]', '', alternate_name) 
        alternate_name_split = set(alternate_name.split())
        if most_common_name_freq < alternate_name_freq:
            # Matching strategy is strict, replace the names.
            if (matching_strategy == 'STRICT'):
                most_common_name = alternate_name_og
            # This can only happen - Honda Accord 2012 vs 2012 Honda Accord
            elif matching_strategy == 'RELAXED' and alternate_name_split == official_name_split:
                most_common_name = alternate_name_og # clean up
            # Penalize shorter words that are subsets in the bigger word, e.g. Honda vs Honda Accord 
            elif matching_strategy == 'RELAXED' and not alternate_name_split.issubset(official_name_split):
                most_common_name = alternate_name_og
            # Update the frequency.
            most_common_name_freq = alternate_name_freq # subset will generally have a higher freq.
                
            
    if official_name_og != most_common_name:
        print('Changing name to', most_common_name, official_name_og)
    return most_common_name




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Args')
    parser.add_argument('--database', type=str, default='LAION400M', help='Laion pretrainig dataset to parse.')
    parser.add_argument('--downstream', type=str, default='imagenet_1k', help='Downstream dataset.')
    parser.add_argument('--datasource', type=str, default='/ssd0/sparasha', help='Location where DB is.')
    parser.add_argument('--max_threads', type=int, default=16, help='Max number of threads to spawn.')
    parser.add_argument('--num_proc', type=int, default=32, help='Number of processes to query Sqlite')
    parser.add_argument('--prefix', type=str, default=None, help='Prefix for datasets like EuroSat.')
    parser.add_argument('--matching_strategy', type=str, default='RELAXED', choices=['RELAXED', 'STRICT'])
    parser.add_argument('--tag', type=str, default=None, )
    args = parser.parse_args()
    
    laion_parser = LaionParser(database=args.database, mode='parallel', 
                               source=args.datasource, 
                               max_threads=args.max_threads, 
                               num_proc=args.num_proc,
                               prefix=args.prefix,
                               matching_strategy=args.matching_strategy)
 
    if args.tag is not None:
        metrics = json.load(open(f'./{args.downstream}/metrics-{args.tag}.json', 'r'))
    else:
        metrics = json.load(open(f'./{args.downstream}/metrics.json', 'r'))
    
    start =time.time()
    mined_captions, metrics, infrequent_classes = laion_parser.get_text_parallel(metrics=metrics)
    print(f'Total time taken: {time.time()-start}')
    file_name_tag = f'{args.database}'
    
    if args.tag is not None:
        file_name_tag += f'-{args.tag}'
    if args.prefix is not None:
        file_name_tag += f'-{args.prefix}'

    with open(f'./{args.downstream}/mined_captions-{file_name_tag}', 'wb') as f:
        pickle.dump(mined_captions, file=f) 
    with open(f'./{args.downstream}/metrics-{file_name_tag}.json', 'w') as f:
        f.write(json.dumps(metrics))
    laion_parser.conn.close()

