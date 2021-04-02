from shutil import copyfile
import glob, sys, os, magic, threading

files = list(set(glob.glob('./images/*.png')))
args = sys.argv[1:]
name = args[0]

for file in files:
    if name in file:
        fmt = file.split('\\')[1]
        if not os.path.isdir('./training_data/'):
            os.mkdir('./training_data/')
        elif not os.path.isdir('./training_data/{}'.format(name)):
            os.mkdir('./training_data/{}'.format(name))
        else:
            f_stat = magic.from_file(file)
            # Both of these might be unnecessary (especially the '128 x 128' since the image will be resized when loaded)
            if '128 x 128' in f_stat and '8-bit/color RGBA' in f_stat:
                # print(f_stat)
                copyfile(file, './training_data/{}/{}'.format(name, fmt))

# Need to get EVERY camelCase/PascalCase word from every emote.
# We could have multiple threads working on multiple parts of the array. For example, divide the array into 8 equal parts and make 8 threads to operate on each separate part.
# We also need to move this project back to the G: disk due to the amount of size this project could take.