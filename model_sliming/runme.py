import os

# split data into train val test
os.system('python3 split_curated.py')

# train a over-sized model
os.system('CUDA_VISIBLE_DEVICES=1 python3 train4prune.py')
# test trained over-sized model
os.system('python3 test_before_prune.py')

# prune model
os.system("python3 cnn_pruning.py")

# finetune pruned model
os.system('CUDA_VISIBLE_DEVICES=1 python3 finetune_pruned.py')
# test_after_pruned
os.system("python3 test_after_pruned.py")
