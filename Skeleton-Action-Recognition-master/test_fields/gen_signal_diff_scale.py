import torch
import art

if __name__ == '__main__':
    art_1 = art.text2art("SKELETON Recognizer",font='cybermedium',chr_ignore=True)
    print(art_1)
    with open('temp.txt', 'w') as f:
        f.write(art_1)
