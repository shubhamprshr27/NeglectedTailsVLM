# datasets=('oxford_pets' 'cub2011' 'flowers102' 'fgvc_aircraft' 'food101' 'stanford_cars' 'imagenet_1k' 'dtd' 'eurosat')
archs=('ViT-B/32' 'ViT-B/16' 'ViT-L/14')
pre_training_datasets=('laion400m' 'laion2b')
for dataset in "${datasets[@]}"; do
    for arch in "${archs[@]}"; do
        for pre_training_corpus in "${pre_training_datasets[@]}"; do
            python pre_extract_features.py --dataset "$dataset" --arch "$arch" --pre_training_corpus "$pre_training_corpus"
        done
    done
done
