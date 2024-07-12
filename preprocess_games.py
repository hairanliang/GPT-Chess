# Open the file for writing

with open('/Users/hairanliang/Documents/Chess.com/MyGames/allPgns copy.txt', 'r') as file:
    filedata = file.read()

filedata = filedata.replace('@', '')

with open('/Users/hairanliang/Documents/Chess.com/MyGames/allPgns copy.txt', 'w') as file:
    file.write(filedata)
