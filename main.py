import os

# clear old figures
os.system("rm -r -f ./figures/")

os.system("mkdir ./figures/")

# create all figures

print("creating figures...")

os.system("python3 ./src/fig1.py")
os.system("python3 ./src/fig2.py")
os.system("python3 ./src/fig3.py")

# run biber and create manuscript twice
for _ in range(2):
    os.system("biber manuscript")




# make auxiluary directory
#os.system("mkdir aux")
# write pdflatex output there
#os.system("pdflatex --output-directory=./aux/ manuscript.tex")
    os.system("pdflatex manuscript.tex")
# move pdf to main directory
#os.system("mv ./aux/manuscript.pdf .")
# remove auxilary directory
#os.system("rm -r -f ./aux/")
os.system("rm ./*.aux")
os.system("rm ./*.bbl")
os.system("rm ./*.bcf")
os.system("rm ./*.blg")
os.system("rm ./*.run.xml")
os.system("rm ./*.out")
os.system("rm ./*.log")


print("done")
