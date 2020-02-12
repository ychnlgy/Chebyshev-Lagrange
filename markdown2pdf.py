# Copied from https://stackoverflow.com/questions/4135344/is-there-any-direct-way-to-generate-pdf-from-markdown-file-by-python

from markdown2 import markdown

input_filename = 'README.md'
output_filename = 'README.html'

with open(input_filename, 'r') as f:
    html_text = markdown(f.read())

with open(output_filename, "w") as f:
    f.write(html_text)

print("Saved to %s." % output_filename)
