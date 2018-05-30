#Parsing script

import os, sys, re

dirname = sys.argv[1]

files = [os.path.join(dirname, name) for name in os.listdir(dirname) if( not "_instructions.txt" in name and not "_ingredients.txt"  in name)]

for f in files:

    html_filename = f

    html_file = open(html_filename)

    instr_outfile_name = html_filename.replace(".html","") + "_instructions.txt"
    instr_outfile = open(instr_outfile_name, "w")
    outfile_name = html_filename.replace(".html","") + "_ingredients.txt"
    outfile = open(outfile_name, "w")

    recipe_name = ""
    line = html_file.readline()
    #Unique string for directions lines
    direction_string = '<span class="recipe-directions__list--item">'
    while line:
        #Get title of recipe
        if("<title>" in line):
            recipe_name = line.replace("<title>", "").replace("</title>","").replace(" - Allrecipes.com","").replace("Recipe","").strip()
            #Getting this one page repeatedly for missing recipes it seems. Want to avoid replicating this one in the database
            if("Three Cheese Italian Style Chicken Sausage Skillet Pizza" in recipe_name):
                outfile.close()
                os.remove(outfile_name)
                os.remove(instr_outfile_name)
                break
            #print(recipe_name)
            outfile.write(recipe_name + "\n")
        #Ingredients
        elif("<label ng-class=" in line and "title=\"" in line):
            ingredient = line[line.find("title=\""):]
            ingredient = ingredient.replace("title=\"", "").replace("\">", "")
            #print(ingredient)
            outfile.write(ingredient)
        #Directions
        elif(direction_string in line):
            direction = line[line.find(direction_string):]
            direction = direction.replace(direction_string,"").strip()
            instr_outfile.write(direction + "\n")

        line = html_file.readline()

    outfile.close()
    instr_outfile.close()
