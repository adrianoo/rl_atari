def log(msg, filename):
    print msg
    with open(filename, 'a') as f:
        f.write(msg)
        f.write('\n')
