import csv
from src.experiment.prepare_data import process_data

with open("/mnt/netstore1_home/aidan.bell/MathStackExchange/MSE.tsv", "r", encoding='utf-8') as f:
    reader = csv.reader(f, delimiter='\t')
    # for i in range(2):
    #     print(next(reader))
    unique_images = 0
    for row in reader:
        if row[1][-2]+row[1][-1] == '_0':
            unique_images+=1
    print(unique_images)

    process_data("/mnt/netstore1_home/aidan.bell/MathStackExchange/MSE.tsv", "/mnt/netstore1_home/aidan.bell/MathStackExchange/MSE_images/MathmaticaImages", validate_data=True)



    
