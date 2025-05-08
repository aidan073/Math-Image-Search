import os
"""
Because I ran filtering on two different servers, the image paths were different and had to be updated.
"""

def update_image_paths(metadata_path, output_path, mse_image_dir, wiki_image_dir):
    updated_rows = []

    with open(metadata_path, 'r', encoding='utf-8') as f:
        for line in f:
            columns = line.strip().split('\t')
            if len(columns) < 3:
                raise IndexError()

            id = columns[0]
            if id[-2:] == '_0': # MSE sample
                new_path = os.path.join(mse_image_dir, f"{id}.png")
            else: # Wikipedia sample
                new_path = os.path.join(wiki_image_dir, id)

            columns[2] = new_path # Update image path
            updated_rows.append('\t'.join(columns))

    with open(output_path, 'w', encoding='utf-8') as f:
        for row in updated_rows:
            f.write(row + '\n')

if __name__ == "__main__":
    update_image_paths("Sim-0.8-Val/Meta.tsv", "Sim-0.8-ValF/Meta.tsv", "/mnt/netstore1_home/aidan.bell/MathStackExchange/MSE_images/MathmaticaImages/", "/mnt/netstore1_home/aidan.bell/WikipediaMath/Wiki_Images/")
