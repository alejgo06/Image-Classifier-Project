#python train.py --data_directory /home/workspace/aipnd-project/flowers -gpu --epochs 2 --save_dir /home/workspace/check1.pth

#predict
#python predict.py -gpu --top_k 5 --path_image '/home/workspace/aipnd-project/flowers/train/10/image_07086.jpg' --path_checkpoint '/home/workspace/checkpointold.pth'

#score
 #python score.py --path_checkpoint check30.pth


python train.py --data_directory /home/workspace/aipnd-project/flowers -gpu --epochs 12 --hidden_units 500 --save_dir /home/workspace/check12

#python train.py --data_directory /home/workspace/aipnd-project/flowers -gpu --epochs 20 --hidden_units 500 --save_dir /home/workspace/check20

#python train.py --data_directory /home/workspace/aipnd-project/flowers -gpu --epochs 30 --hidden_units 500 --save_dir /home/workspace/check30

python train2.py --data_directory /home/workspace/aipnd-project/flowers -gpu --epochs 30 --hidden_units 500 --save_dir /home/workspace/check30


