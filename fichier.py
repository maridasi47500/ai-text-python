import os
import json
class Fichier:
  def __init__(self,path,name):
    self.path=path
    self.name=name
  def lire(self):
    print(self.path+"/"+self.name)
    j=open(self.path+"/"+self.name, "r")
    return j.read()
  def ligneparligne(self):
    j=open(self.path+"/"+self.name, "r")
    return j.readlines()
  def liretousfichiers(self):
    list=os.listdir(self.path)
    mytext=""
    for yeah in list:
      j=open(self.path+"/"+yeah, "rb")
      mytext+=j.read().decode()
    return mytext.replace("\n","<br>")
  def lirefichier(self):
    print(self.path+"/"+self.name)
    j=open(self.path+"/"+self.name, "rb")
    return j.read()
  def ecrire(self,mycontent):
    hey=open((self.path+"/"+self.name),"w")
    hey.write(mycontent)
    hey.close()
  def ecrirejson(self,data):
    with open((self.path+"/"+self.name), 'w') as json_file:
       json.dump(data, json_file, indent=4)
  def lirejson(self):
    original_json=None
    with open((self.path+"/"+self.name), mode="r", encoding="utf-8") as input_file:
       original_json = input_file.read()
    return original_json

