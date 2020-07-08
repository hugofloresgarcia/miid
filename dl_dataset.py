"""
download philharmonia samples from
https://philharmonia.co.uk/resources/sound-samples/
"""
import urllib.request
import zipfile, re, os
import pandas as pd

def extract_nested_zip(zippedFile, toFolder):
    """ Extract a zip file including any nested zip files
        Delete the zip file(s) after extraction
    """
    with zipfile.ZipFile(zippedFile, 'r') as zfile:
        zfile.extractall(path=toFolder)
    os.remove(zippedFile)
    
    for root, dirs, files in os.walk(toFolder):
        dirs[:] = [d for d in dirs if not d[0] == '_']
        for filename in files:
            if re.search(r'\.zip$', filename):
                fileSpec = os.path.join(root, filename)
                print(f'extracting: {fileSpec}')
                filename = filename[0:-4]
                extract_nested_zip(fileSpec, os.path.join(root, filename))
                
def generate_dataframe(root_dir):
    """
    generate a dictionary for metadata from our dataset
    """
    data = []
    for root, dirs, files in os.walk(root_dir):
        for d in dirs:
            generate_dataframe(os.path.join(root, d))
        # we just want mp3s

        for f in files:
            if f[-4:]  == '.mp3':
                fsplit = f.split('_')
                metadata = {
                    'instrument': fsplit[0],
                    'pitch': fsplit[1], 
                    'path_to_audio': os.path.join(root, f)
                }
                data.append(metadata)
    return pd.DataFrame(data)


if __name__ == "__main__":
    
    url = "https://philharmonia-assets.s3-eu-west-1.amazonaws.com/uploads/2020/02/12112005/all-samples.zip"
    save_path = "./data/philharmonia"
    
    # download the zip
    print(f'downloading from {url}')
    print('this may take a while ? :/ i think its like 250MB')
    urllib.request.urlretrieve(url, f'{save_path}.zip')
    
    # extract everything recursively
    extract_nested_zip(f'{save_path}.zip', save_path)
    
    print('generating dataframe...')
    df_path = os.path.join(save_path, 'all-samples', 'metadata.csv')
    df = generate_dataframe(os.path.join(save_path, 'all-samples'))
    df.to_csv(df_path)
    print(f'dataframe saved to {df_path}')
    print('all done!')
    
    
