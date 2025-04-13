import os
import gdown
import zipfile

class CategorizedCorpus:
    def __init__(self, path):
        self.path = path
        self.folders = [f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))]

    def __iter__(self):

        for category in self.folders:

            dir_path = os.path.join(self.path, category)

            for file in os.listdir(dir_path):

                file_path = os.path.join(dir_path, file)

                if os.path.isfile(file_path):

                    with open(file_path, 'r', encoding='utf-8') as file:
                        text = file.read().strip()
                        yield text, category
        
