###################### ADD TO REST ######################
def adds_in_rst(filehandle):
    # Write carbon adds
    filehandle.write("\n\n.. raw:: html\n")
    filehandle.write("\n   <hr>")
    filehandle.write("\n   <center>")
    filehandle.write('\n     <script async type="text/javascript" src="//cdn.carbonads.com/carbon.js?serve=CEADP27U&placement=erdogantgithubio" id="_carbonads_js"></script>')
    filehandle.write("\n   </center>")
    filehandle.write("\n   <hr>")

###################### SCAN DIRECTORY ######################
def scan_directory(currpath, directory, ext):
    # Uitlezen op ext
    path_to_files = os.path.join(currpath, '_static', directory)
    files_in_dir = np.array(os.listdir(path_to_files))
    Iloc = np.array(list(map(lambda x: x[-len(ext):]==ext, files_in_dir)))
    return files_in_dir[Iloc]

###################### EMBED PDF IN RST ######################
def embed_in_rst(currpath, directory, ext, title, file_rst):

    try:
        # Uitlezen op extensie
        files_in_dir = scan_directory(currpath, directory, ext)
        print('---------------------------------------------------------------')
        print('[%s] embedding in RST from directory: [%s]' %(ext, directory))
    
        # Open file
        filehandle = open(file_rst, 'w')
        filehandle.write(".. _code_directive:\n\n" + title + "\n#######################\n\n")
    
        # 3. simple concat op 
        for fname in files_in_dir:
            print('[%s] processed in rst' %(fname))
            title = fname[:-len(ext)] + '\n' +  '*'*len(fname) + "\n"
            if ext=='.pdf':
                newstr = ":pdfembed:`src:_static/" + directory + "/" + fname + ", height:600, width:700, align:middle`"
            elif ext=='.html':
                newstr = ".. raw:: html\n\n" + '   <iframe src="_static/' + directory + "/" + fname + '"' + ' height="900px" width="750px", frameBorder="0"></iframe>'
            write_to_rst = title + "\n" + newstr + "\n\n\n\n"
            # Write to rst
            filehandle.write(write_to_rst)
	
	    # ADDs in RST wegschrijven
        adds_in_rst(filehandle)
        # Close file
        filehandle.close()
    except:
        print('ERROR IN EMBEDDING IN RST.')


###################### CONVERT NOTEBOOKS TO HTML ######################
def convert_ipynb_to_html(currpath, directory, ext):
    try:
        # Uitlezen op extensie
        files_in_dir = scan_directory(currpath, directory, ext)
        # 3. simple concat op 
        for fname in files_in_dir:
            path_to_file = os.path.join('_static/', directory, fname)
            print('[%s] converting to HTML' %(path_to_file))
            os.system('jupyter nbconvert --to html ' + path_to_file)
    except:
        print('ERROR IN CONVERTING NOTEBOOK TO HTML.')


# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import numpy as np
import os
import sys
sys.path.insert(0, os.path.abspath('../../'))
currpath = os.path.dirname(__file__)
import pca

# -- Import PDF from directory in rst files -----------------------------------
embed_in_rst(currpath, 'pdf', '.pdf', "Additional Information", 'Additional_Information.rst')

# -- Import notebooks in HTML format -----------------------------------------
convert_ipynb_to_html(currpath, 'notebooks', '.ipynb')
embed_in_rst(currpath, 'notebooks', '.html', "Notebook", 'notebook.rst')


# -- Project information -----------------------------------------------------

project = 'pca'
copyright = '2022, Erdogan Taskesen'
author = 'Erdogan Taskesen'

# The master toctree document.
master_doc = 'index'

# The full version, including alpha/beta/rc tags
release = 'pca'
version = str(pca.__version__)

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
	"sphinx.ext.intersphinx",
	"sphinx.ext.autosectionlabel",
	"rst2pdf.pdfbuilder",
	"sphinxcontrib.pdfembed",
#    'sphinx.ext.duration',
#    'sphinx.ext.doctest',
#    'sphinx.ext.autosummary',
]


napoleon_google_docstring = False
napoleon_numpy_docstring = True

# autodoc_mock_imports = ['cv2','keras']


pdf_documents = [('index', u'pca', u'pca', u'Erdogan Taskesen'),]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build"]


# The name of the Pygments (syntax highlighting) style to use.
pygments_style = "sphinx"

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
# html_theme = 'alabaster'
# html_theme = 'default'
html_theme = 'sphinx_rtd_theme'


# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# These paths are either relative to html_static_path
# or fully qualified paths (eg. https://...)
html_css_files = ['css/custom.css',]

# A list of files that should not be packed into the epub file
epub_exclude_files = ['search.html']

# -- Options for EPUB output
epub_show_urls = 'footnote'

# html_sidebars = { '**': ['globaltoc.html', 'relations.html', 'carbon_ads.html', 'sourcelink.html', 'searchbox.html'] }




