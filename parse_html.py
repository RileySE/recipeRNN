#Parsing script

import os, sys, re

dirname = sys.argv[1]

include_instructions = False

files = [os.path.join(dirname, name) for name in os.listdir(dirname)]

for f in files:

    html_filename = f

    html_file = open(html_filename)

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
        elif(include_instructions and direction_string in line):
            direction = line[line.find(direction_string):]
            direction = direction.replace(direction_string,"").strip()
            outfile.write(direction + "\n")

        line = html_file.readline()

    outfile.close()
    
