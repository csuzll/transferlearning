import os

def ListFilesToTxt(dir, file, wildcard, recursion):
    exts = wildcard.split(" ")
    files = os.listdir(dir)
    for name in files:
        fullname=os.path.join(dir,name)
        if(os.path.isdir(fullname) & recursion):
            ListFilesToTxt(fullname,file,wildcard,recursion)
        else:
            for ext in exts:
                if(name.endswith(ext)):
                    (filename,extension) = os.path.splitext(name)
                    file.write(fullname+"\n")
                    break

if __name__ == '__main__':
    dirpath="F:\\data\\chest_pro_data\\normal"
    outfile="F:\\data\\chest_pro_data\\normal.txt"
    wildcard = ".jpg"

    file = open(outfile,"w")
    if not file:
        print ("cannot open the file %s for writing" % outfile)
    ListFilesToTxt(dirpath,file,wildcard, 1)

    file.close()