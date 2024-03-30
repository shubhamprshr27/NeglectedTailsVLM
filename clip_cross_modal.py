import torch
from utils.extras import get_engine, OPENCLIP_MODEL_DIC
from torch.utils.data import DataLoader
from models import MyLinear
from utils import features
from testing import validate
from utils.optimizers import get_optimizer, get_warmup_scheduler
from utils.datasets.tensor_dataset import TensorDataset, TextTensorDataset

criterion = torch.nn.CrossEntropyLoss()
DEVICE = "cpu" if torch.cuda.is_available() else "cpu"


"""
    Code inspired from: https://github.com/linzhiqiu/cross_modal_adaptation
"""
def cross_modal_train(model,head, optimizer, scheduler, tokenizer, prompts_dataloader, 
                      mined_dataloader, val_dataloader, logger, device='cuda', 
                      logit_scale=4.6017, zero_shot_weights=None, max_iters=19200, wise_ft_alpha=0.5):
    best_head = None
    if mined_dataloader:
        image_loader_iter = iter(mined_dataloader)
    else:
        image_loader_iter = None

    if prompts_dataloader:
        text_loader_iter = iter(prompts_dataloader)
    else:
        text_loader_iter = None

    train_acc = 0
    train_count = 0
    logit_scale = torch.tensor([logit_scale]).to(device=device)

    for (i) in range(max_iters):
        model.eval()
        head.train()

        if image_loader_iter:
            try:
                imgs, img_labels = next(image_loader_iter)
            except StopIteration:
                image_loader_iter = iter(mined_dataloader)
                imgs, img_labels = next(image_loader_iter)
            img_feats = imgs.to(device)
            img_feats_norm = img_feats / img_feats.norm(dim=-1, keepdim=True)
        else: 
            img_feats = None

        if text_loader_iter:
            try:
                text_feats, text_labels = next(text_loader_iter)
            except StopIteration:
                text_loader_iter = iter(prompts_dataloader)
                text_feats, text_labels = next(text_loader_iter)
            text_feats = text_feats.to(device)
            text_feats_norm = text_feats / text_feats.norm(dim=-1, keepdim= True)
        else:
            text_feats = None

        if image_loader_iter is not None and text_loader_iter is not None:
            features = torch.cat([img_feats_norm, text_feats_norm], dim=0)
            labels = torch.cat([img_labels, text_labels], dim=0)
        elif image_loader_iter is not None:
            features = img_feats_norm
            labels = img_labels
        elif text_loader_iter is not None:
            features = text_feats_norm
            labels = text_labels
        else:
            raise ValueError('Training without Images and Text.')

        
        labels = labels.to(device).long()

        logits = head(features) 
        logits = logits * logit_scale.exp()
        loss = criterion(logits, labels)
        pred = torch.argmax(logits, dim=1)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        train_acc += torch.sum(pred == labels).item()
        train_count += labels.size(0)
    best_head = get_wise_ft_head(alpha=wise_ft_alpha, 
                                 head=head, 
                                 zero_shot_weights=zero_shot_weights, 
                                 device=device)
    val_acc, val_confusion_mat = validate(val_dataloader, 
                                          model, 
                                          logger=logger, 
                                          classifier_head=best_head, 
                                          show_confusion_matrix=True, 
                                          Epoch=i, 
                                          device = device,
                                          pre_extracted=True)           
    return val_acc, best_head.cpu(), val_confusion_mat

def get_wise_ft_head(alpha=0.5, head=None, zero_shot_weights=None, device='cuda'):
    new_weights = alpha * head.linear.weight.data.to(device) + (1 - alpha) * zero_shot_weights.to(device)
    return MyLinear(weights=new_weights, bias=False).to(device)

def get_exp_name(params_dict):
    return '_'.join([f"{k}_{v}" for k, v in params_dict.items()])


def train(arch, 
          pre_training_corpus, 
          prompts, 
          shots=100,
          wise_ft_alpha = 1.0, 
          logit_scale=4.60517, 
          extracted_feats_path = None,
          bsz=64,
          lr=5e-4,
          wd=0,
          tags=None, 
          dataset='imagenet_1k', 
          max_iters=32000):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, val_preprocess, tokenizer = get_engine(arch=arch, mode='train', corpus=pre_training_corpus)
    model.float()
    model.cuda()
    openclip_arch_name = OPENCLIP_MODEL_DIC[pre_training_corpus][arch][1]
    
    # Prompts and zero shot classifier.
    prompt_tensors = features.get_text_features(model, prompts, logger=None, tokenize = tokenizer)
    zeroshot_weights = features.prompt_sampler(prompt_tensors, logger=None, sample_by='mean')
    print('Made prompt tensors.', zeroshot_weights.shape)
    head = MyLinear(weights=zeroshot_weights, bias=False)
    head.to(device=device)

    optimizer = get_optimizer(head.parameters(), optim_type = 'AdamW', lr = lr, wd = wd)

    base_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(max_iters))
    if lr > 5e-5:
        warmup_lr = 1e-5
    else:
        warmup_lr = 5e-6
    
    scheduler = get_warmup_scheduler(optimizer=optimizer, scheduler=base_scheduler, warmup_iter=50, warmup_lr=warmup_lr)
    
    
    val_dataset = TensorDataset(dataset_root='./data', dataset=dataset, arch=openclip_arch_name, pre_trained_corpus=pre_training_corpus)
    val_dataloader = DataLoader(val_dataset, batch_size=4096, shuffle=False, num_workers=0, drop_last=False)
    
    # Mined dataset from split.
    mined_dataset = TensorDataset(dataset_root='./data', 
                                  dataset=f'{dataset}_mined', 
                                  arch=openclip_arch_name, 
                                  pre_trained_corpus=pre_training_corpus, 
                                  shots=shots, 
                                  split='mined', 
                                  base_path = extracted_feats_path,
                                  tags=tags)
    mined_dataloader = DataLoader(mined_dataset, 
                                  batch_size=bsz, 
                                  shuffle=True, 
                                  num_workers=0,
                                  drop_last=True)
    text_dataset = TextTensorDataset(model=model, tokenizer=tokenizer, prompts=prompts)
    text_dataloader = DataLoader(text_dataset, batch_size=bsz, shuffle=True, pin_memory=True, drop_last=True, num_workers=0)

    zs_val_acc, zs_confusion_mat = validate(val_dataloader, model, logger=None, classifier_head=head, show_confusion_matrix = True, Epoch=-1, device = device, pre_extracted=True)
    print('ZS accuracy:',zs_val_acc )
    best_val_acc, best_head, val_confusion_mat = cross_modal_train(model=model, 
            head=head, 
            optimizer=optimizer, 
            scheduler=scheduler,
            tokenizer=tokenizer,
            prompts_dataloader= text_dataloader,
            mined_dataloader = mined_dataloader,
            val_dataloader=val_dataloader,
            logger=None,
            device=device,
            logit_scale=logit_scale,
            zero_shot_weights=zeroshot_weights,
            max_iters=max_iters,
            wise_ft_alpha=wise_ft_alpha, # 0.5
    )
    print('Testing Acc:',best_val_acc)
    return best_val_acc, best_head, [zs_confusion_mat,val_confusion_mat]