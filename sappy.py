from functools import reduce
from sys import exit
# for downloading license files from Github repositories
import requests
# for string substitution
from re import sub, findall
# for listing directories and files
from os import listdir
# for `sqrt` function
from math import sqrt
# for argument parsing
from argparse import ArgumentParser
# for stemming and tokenizing words
from nltk.stem.porter import PorterStemmer
# for computing TF-IDF (document X features) matrix
from sklearn.feature_extraction.text import TfidfVectorizer


def prepare_license_file(filename):
    """
    Loads and cleans license text from the file.
    /
    :filename: a path to the license's file
    :type filename: string
    /
    Returns cleaned version of license's text. Type: string.
    Used only once in `load_license_templates` method.
    """
    with open(filename, 'r') as f:
        lines = f.readlines()
    # Join all lines in one single string
        raw_text = reduce((lambda s1, s2: s1 + s2), lines)
        return clean(raw_text)


def clean(document):
    """
    Cleans 'document': stems and lowers each token, removes all punctuation.
    /
    :document: a string to be cleaned
    :type document: string
    /
    Returns prepared version of 'document'. Type: string.
    /
    We need to prepare text of the license before doing anything with it, so
    it's userful to have this method.
    """
# Delete all useless punctuation signs
    document = sub('[^\w\d ]', ' ', document)
# Only single space is allowed between words
    document = document.replace('  ', ' ').lower()  # explicit case lowering
    stemmer  = PorterStemmer()
# Stem and join togehter all tokens separated by space
    cleaned_document = reduce(
            lambda acc, token: acc + ' ' + stemmer.stem(token),
            document.split(' '),
            '')
    return cleaned_document.strip()


def load_license_templates():
    """
    Returnes list of cleaned license templates' texts and its' names.
    Used only once in `detect_license` method.
    """
    TEMPLATES_DIR     = 'license-templates/'
    license_filenames = [name for name in listdir(TEMPLATES_DIR) if
                            name.endswith('.txt')]
    names    = []
    licenses = []
    for lic_fn in license_filenames:
        names.append(sub('-', ' ', lic_fn[:-4].upper()))
        licenses.append(prepare_license_file(TEMPLATES_DIR + lic_fn))
    return names, licenses


def dot(v, w):
    """
    Computes the dot product of two vectors stored in Compressed Sparse Row
    format.
    /
    :v: a float vector.
    :type v: Compressed Sparse Row Matrix.
    /
    :w: a float vector.
    :type w: Compressed Sparse Row Matrix.
    /
    Returns a float number.
    """
    return (v * w.T)[0, 0]


def cosine_similarity(v, w):
    """
    Computes a cosine of an angle between two vectors `v` and `w`.
    /
    :v: a float vector.
    :type v: Compressed Sparse Row Matrix.
    /
    :w: a float vector.
    :type w: Compressed Sparse Row Matrix.
    /
    Returns a float number.
    """
    dist_v = sqrt(dot(v, v))
    dist_w = sqrt(dot(w, w))
    return dot(v, w) / (dist_v * dist_w)


def detect_license(document, cleaned=False):
    """
    Finds a license that is most similar to the provided `document`.
    /
    :document: a license, whose name should be identified
    :type document: string
    /
    :cleaned: shows whether a `document` is prepared for vectorization.
    /
    Returns the name of the license document in string.
    """
    vectorizer = TfidfVectorizer(stop_words='english',
                                  strip_accents='unicode',
                                  use_idf=True,
                                  smooth_idf=True,
                                  norm='l2')
    names, licenses = load_license_templates()
# `tfidf` is a docXvocab matrix, where each row is a document and each
# column is a token in vocabulary
    cleaned_doc = document if cleaned else clean(document)
    tfidf       = vectorizer.fit_transform(licenses + [cleaned_doc])
# Last row in this matrix is our `document`
    vectorized_document           = tfidf[-1]
    index_of_most_similar_license = 0
    max_similarity                = -1
# Searching for most similar license
    for i in range(0, len(licenses)):
        next_license = tfidf[i]
        cos          = cosine_similarity(vectorized_document, next_license)
        if cos > max_similarity:
            max_similarity                = cos
            index_of_most_similar_license = i
    return names[index_of_most_similar_license]


def parse_args():
    """
    Parses command-line arguments and returns username, title of specified
    repository and its' branch.
    Returns: tuple (username, repo_name, branch).
    Used only once in `main` method.
    """
    DESC = 'Automatic license detection of a Github repository.'
    parser = ArgumentParser(description=DESC)
# Specify agruments
    parser.add_argument('--branch',
                        default='master',
                        required=False,
                        help='A branch of a repository from which license file should be obtained. Default: `master`.')
    parser.add_argument('--repository_name',
                        required=False,
                        help='A name of a repository, whose license needs to be detected. Required.')
    parser.add_argument('--username',
                            required=False,
                            help='A name of a user who owns a repository. Required.')
    parser.add_argument('--url',
                            required=False,
                            help='An URL to Github repository.')
# Start parsing sys.argv
    arg_dict = parser.parse_args().__dict__
    branch   = arg_dict['branch']  # `master` by default
    user     = arg_dict['username']
    repo     = arg_dict['repository_name']
    url      = arg_dict['url']
    if (user is None) or (repo is None):
        if (url is None):
            # No repository information was typed, exiting...
            print('Usage: --user <USERNAME> --repo <REPOSITORY NAME> --branch'
                    '<BRANCH NAME> (optional) or ')
            print('--url <LINK TO REPOSITORY>')
            exit(-1)
        # Cut the `http` header of an URL
        chopped_url = sub('https+:\/\/', '', url)
        # Extract user and repository names from URL
        user, repo = findall('\/{1}([^\/]+)', chopped_url)
    return user, repo, branch


def main():
    """
    Parses and executes specified command.
    """
    user, repo, branch = parse_args()
    url      = 'https://raw.github.com/%s/%s/%s/' % (user, repo, branch)
    blob_url = 'https://raw.github.com/%s/%s/blob/%s/' % (user, repo, branch)
    names    = ['LICENSE', 'LEGAL', 'COPYRIGHT', 'COPYING']
    formats  = ['', '.md', '.txt']
    license_files = [name + format for name in names for format in formats]
    license_detected = False
    print('Arguments parsed. Searching for a license file in a %s/%s' % (user, repo))
# Trying to download license file from specified repository
    for fname in license_files:
        try:
            r1      = requests.get(url + fname)
            r2_blob = requests.get(blob_url + fname)
        except:
            print('Unable connect to the Internet.')
            return
        request = r1 if r1.ok else r2_blob
        if request.ok:
            print('License file found. Trying to detect it''s name.')
            license          = detect_license(request.text)
            license_detected = True
            print('This repository is probably licensed under %s license.' % license)
            break
    if not license_detected:
        print('License file not found in specified repository.')


if __name__ == '__main__':
    print('Modules loaded.')
    main()
