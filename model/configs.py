def Bin_config():
    config = {}

    config['batch_size'] = 2048
    config['num_workers'] = 0
    config['epochs'] = 30
    config['lr'] = 0.01
  
    config['model'] = 'gru'        # cnn, lstm, gru, transformer, cnn_maxpool


    if config['model'] == 'lstm' or config['model'] == 'gru':
        config['flatten_dim'] = 1216
    elif config['model'] == 'cnn':
        config['flatten_dim'] = 64
    elif config['model'] == 'cnn_maxpool':
        config['flatten_dim'] = 12
    elif config['model'] == 'transformer':
        config['flatten_dim'] = 38
    return config
