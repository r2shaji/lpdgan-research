# lpdgan-research

Dataset link - https://uofwaterloo-my.sharepoint.com/:u:/g/personal/r2shaji_uwaterloo_ca/EYqxG5vIxH1GgzeXn9MJsNwBXKbf_aBm_Ch0pzJFfHbyPg  - Copy the dataset folder contents to the dataset folder in the code

Venv link - https://uofwaterloo-my.sharepoint.com/:u:/g/personal/r2shaji_uwaterloo_ca/EYrwbqyn81VNn1YQljam_bcBIWxjdDagPTlk_Wrb8b2rvw?e=8oUyrF - Copy the venv folder to the code and run it 

or ``` pip install -r requirements.txt``` to create the environment.

You can train the model with ```python main.py --mode train --dataroot "\path to folder\dataset\"```. The checkpoints will be stored in checkpoints folder. This folder will contain an LPDGAN folder which will have a log_loss.txt which stores the training loss information.

![image](https://github.com/user-attachments/assets/8263f0b8-87e8-4412-9cfd-a727da3e30e8)

![image](https://github.com/user-attachments/assets/cdfdca49-d7f3-4b37-9778-180eefe5cb14)



You can test the model with ```python main.py --mode test --dataroot "\path to folder\dataset\"```. The results will be stored in the results folder. The folder will have a test_200_iter200 folder, which will contain an index.html file, which you can open in chrome to see the test results.

![image](https://github.com/user-attachments/assets/8e6c610b-5f64-4d3a-8b6b-e80ad0108636)


