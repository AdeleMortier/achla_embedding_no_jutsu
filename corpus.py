"""A few files from the Knesset dataset could not be parsed properly.
We suspect that it is because then do not contain enough closing tags.
Since those problematic files do not represent a huge fraction of the
total dataset,  we preferred to set them aside instead of trying to
edit them."""
  
def parse_corpus_file(fpath, problematic_files):
    if fpath in ['./16/'+pf for pf in problematic_files]:
        return [], []
    ns, vs = [], []
    tree = ET.parse(fpath)
    root = tree.getroot()
    #print(root)
    for paragraph in root[0]:
        for sentence in paragraph:
            for token in sentence:
                #print(token.attrib['surface'])
                for analysis in token:
                    #print(analysis.attrib['score'])
                    if 'score' in analysis.attrib.keys():
                        if float(analysis.attrib['score']) > 0:
                            for base in analysis:
                                #if 'dottedLexiconItem' in base.attrib.keys():
                                #print(base.attrib['dottedLexiconItem'])
                                if 'lexiconItem' in base.attrib.keys():
                                #print(base.attrib['lexiconItem'])
                                    for pos in base:
                                        if pos.tag == 'verb':
                                            vs.append(base.attrib['lexiconItem'])
                                        if pos.tag == 'noun':
                                            ns.append(base.attrib['lexiconItem'])
    return ns, vs




def write_list_to_file(fname, l):
    with open(f'./{fname}', 'w') as f:
        for x in l:
            f.write(x+'\n')

def load_list_from_file(fname):
    with open(f'./{fname}', 'r') as f:
        l = f.read().split('\n')[:-1]
    return l


def create_wiki_corpus(url, language):
    """ for hebrew: https://dumps.wikimedia.org/hewiki/latest/hewiki-latest-pages-articles.xml.bz2"""
    fname = url.split('/')[-1]
    if not fname in os.listdir('./corpora/'):
        response = wget.download(url, fname)


    if f"wiki.{language}.{model}.text" not in os.listdir('./corpora/'):
        inp = fname
        outp = f"wiki.{language}.{model}.text"
        
        i = 0
        print("Starting to create wiki corpus")
        output = open(outp, 'w')
        space = " "

        wiki = WikiCorpus(inp, dictionary={})

        for i, text in enumerate(wiki.get_texts()):
	    article = space.join(text)
	    output.write("{}\n".format(article))
            if (i % 10000 == 0):
                print("Saved " + str(i) + " articles")

        output.close()
        end_time = time.time()

        print(f"Saved {i} articles in {(end_time-start_time)/60.} minutes")
        
