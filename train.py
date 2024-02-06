import torch
import torch.nn as nn

from dataset import get_dataset
from model import memesnet
from utils import latest_weights_file_path, get_weights_file_path, load_config, Time

def validation_step(config,model,dataset,device):
        model.eval()
        batch = next(iter(dataset['val_dataset']))

        txt_encoder_input = batch['txt_encoder_input'].to(device)
        img_encoder_input = batch['img_encoder_input'].to(device)
        text_encoder_mask = batch['text_encoder_mask'].to(device)
        img_encoder_mask = batch['img_encoder_mask'].to(device)
        label = batch['label']

        with torch.no_grad():
                if torch.cuda.device_count() > 1:
                        out = model.module(x=txt_encoder_input, y=img_encoder_input, text_mask=text_encoder_mask, img_mask=img_encoder_mask) # (batch, num_classes)
                else:
                        out = model(x=txt_encoder_input, y=img_encoder_input, text_mask=text_encoder_mask, img_mask=img_encoder_mask) # (batch, num_classes)
        out = config['classes'][torch.argmax(out.squeeze(0))]
        actual = config['classes'][torch.argmax(label.squeeze(0))]
        print(f'>> Prediction: "{out}", Actual label: "{actual}"')
        
        return 

def train_model(config):
        dataset = get_dataset(config)        
        params = {
                'vocab_size': dataset['tokenizer'].get_vocab_size(),
                'seq_len': config['seq_len'],
                'n_single':config['n_single'],
                'n_cross':config['n_cross'],
                'd_model':config['d_model'],
                'image_features':config['image_features'], 
                'dropout':config['dropout'], 
                'head':config['head'],
                'num_classes':config['num_classes']
        }

        print('\nHardware setup..')
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        device_name = torch.cuda.get_device_name(device.index) if device.type == 'cuda' else 'CPU'
        print('>> Detected device:', device_name)

        model = memesnet(**params).to(device)
        # changing model configuration if multiple GPU avialable
        if torch.cuda.device_count() > 1:
                print(">> Using", torch.cuda.device_count(), "GPUs!")
                model = nn.DataParallel(model)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], eps=1e-9)
        loss_function = nn.CrossEntropyLoss(label_smoothing=config['label_smoothing']).to(device) 
        
        model_file = latest_weights_file_path(config) if config['preload']=='latest' else get_weights_file_path(config, config['preload']) if config['preload'] else None        
        init_epoch = 0
        print(f'\nModel Training on {device}..')
        if model_file:
                state = torch.load(model_file)
                init_epoch = state['epoch']+1
                print(f'>> Resuming model training from epoch no. {init_epoch}')
                model.load_state_dict(state['model'])
                optimizer.load_state_dict(state['optimizer'])
        else:
                print('>> No model to preload, starting from scratch.')

        # training loop
        if torch.cuda.is_available(): 
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
        else:
                tm = Time()
                tm.start(message='training model')
        for epoch in range(init_epoch, config['epochs']):
                torch.cuda.empty_cache()
                model.train()
                batch_count = 0
                if config['batch_data_while_training']:
                        print(f'\n>>> epoch: {epoch+1}')
                for batch in dataset['train_dataset']:
                        txt_encoder_input = batch['txt_encoder_input'].to(device)
                        img_encoder_input = batch['img_encoder_input'].to(device)
                        text_encoder_mask = batch['text_encoder_mask'].to(device)
                        img_encoder_mask = batch['img_encoder_mask'].to(device)
                        label = batch['label'].to(device)

                        if torch.cuda.device_count() > 1:
                                out = model.module(x=txt_encoder_input, y=img_encoder_input, text_mask=text_encoder_mask, img_mask=img_encoder_mask) # (batch, num_classes)
                        else:
                                out = model(x=txt_encoder_input, y=img_encoder_input, text_mask=text_encoder_mask, img_mask=img_encoder_mask) # (batch, num_classes)

                        # loss
                        # print(out.contiguous().dtype, label.dtype, out.contiguous(), label)
                        loss = loss_function(out.contiguous(), label)
                        loss.backward()

                        # optimization
                        optimizer.step()
                        optimizer.zero_grad(set_to_none=True)

                        batch_count+=1
                        if config['batch_data_while_training']:
                                print(f'>>>>> batch: {batch_count}, loss: {loss.item():6.3f}')
                        # break

                # to save the model instance at end of every epoch
                file = get_weights_file_path(config,f'{epoch:03d}')
                torch.save({
                        'epoch': epoch,
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                }, f = file)
                if not config['batch_data_while_training']:
                        print(f'\n>>> epoch: {epoch+1}, loss: {loss.item():6.3f}')

                # validation step
                if config['validation_step_while_training']:
                        vtm = Time()
                        vtm.start(message='validation step')
                        if epoch % config['validation_step_frequency']==0:
                                print('>> Validation step..')
                                validation_step(config,model,dataset,device)
                        vtm.end()
                # print(type(model.parameters()))
        print()
        if torch.cuda.is_available():
                end.record()
                torch.cuda.synchronize()
                print(f'\nTraining complete with total time {start.elapsed_time(end)/1000:.3f} sec.')
        else:
                tm.end()
                print(f'\nTraining complete.')
        return 

def main(config):
        train_model(config)

if __name__=='__main__':
        config = load_config('config.yml')
        main(config)