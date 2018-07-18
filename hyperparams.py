'''
modified from
https://www.github.com/kyubyong/tacotron
'''
class Hyperparams:
    '''Hyper parameters'''

    # where you should not modified
    # where you should not modified
    # where you should not modified
    prepro = True # This should be true  # run `python prepro.py` before running `python train.py`.
    vocab = "PES abcdefghijklmnopqrstuvwxyz'.?" # P: Padding E: End of Sentence S: Start token

    # data path
    prepro_path = "./tacotron_las_dataset/prepro_data" # preprocessed mel and mag
    data = "./tacotron_las_dataset"
    #test_data = 'test_sentences.txt' #for synthesize
    #ref_wavfile = "./VCTK/wavs_trimmed/p225_001.wav" #for synthesize #reference audio
    train_data_name = "transcript_training_tacotron_seqlen50.txt"
    test_data_name = "transcript_testing.txt"
    #train data should be put in the $data/$train_data_name
    # for evaluate las #test data should be put in the $data/$test_data_name

    # signal processing
    sr = 16000 # Sample rate.
    n_fft = 2048 # fft points (samples)
    frame_shift = 0.0125 # seconds
    frame_length = 0.05 # seconds
    hop_length = int(sr*frame_shift) # samples.
    win_length = int(sr*frame_length) # samples.
    n_mels = 80 # Number of Mel banks to generate
    power = 1.2 # Exponent for amplifying the predicted magnitude
    n_iter = 300 # Number of inversion iterations #griffin-lim
    preemphasis = .97 # or None
    max_db = 100
    ref_db = 20

    # model
    # tacotron-1
    embed_size = 256 # alias = E
    encoder_num_banks = 16
    decoder_num_banks = 8
    num_highwaynet_blocks = 4
    r = 5 # Reduction factor. Paper => 2, 3, 5
    dropout_rate = .5
    ## reference encoder
    ref_enc_filters = [32, 32, 64, 64, 128, 128]
    ref_enc_size = [3,3]
    ref_enc_strides = [2,2]
    #ref_enc_gru_size = 128
    ref_enc_gru_size = embed_size

    #model
    #las
    las_embed_size = 256
    hidden_units = 512
    attention_hidden_units = 512
    las_dropout_rate = 0.1 # rate=0.1 would drop out 10% of input units.
    attention_mechanism='original' # original,dot (indicate using original attention or dot attention)

    # training parameter
    batch_size = 32
    taco_consis_weight = 1 # determine how much would attention consistency influence

    # tacotron parameters
    taco_num_epochs=300 #tacotron:300  #las:100
    taco_lr = 0.001 # Init learning rate.  # for training tacotron las
    taco_diff_lr = 0.0001 # for training with unpaired data
    taco_logdir = "./pretrained_tacotron"
    taco_logfile="log"
    #taco_sampledir = 'samples_tacotron' # generate sample dir

    #las parameters (no need to modify when training tacotron)
    #las_num_epochs=100 #tacotron:300  #las:100
    #las_lr = 0.001 # Init learning rate.  # for train tacotron las
    #las_lr_decay=0.9 #decay rate
    las_logdir = "./pretrained_las"
    #las_logfile="log"
    # evaluate las batch size
    #las_inference_batch_size = 300 #how many batch per time

