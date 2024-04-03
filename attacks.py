
import random
from nltk.tokenize import sent_tokenize
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer

human_abstract1 = "A fully differential calculation in perturbative quantum chromodynamics is presented for the production of massive photon pairs at hadron colliders. All next-to-leading order perturbative contributions from quark-antiquark, gluon-(anti)quark, and gluon-gluon subprocesses are included, as well as all-orders resummation of initial-state gluon radiation valid at next-to-next-to-leading logarithmic accuracy. The region of phase space is specified in which the calculation is most reliable. Good agreement is demonstrated with data from the Fermilab Tevatron, and predictions are made for more detailed tests with CDF and DO data. Predictions are shown for distributions of diphoton pairs produced at the energy of the Large Hadron Collider (LHC). Distributions of the diphoton pairs from the decay of a Higgs boson are contrasted with those produced from QCD processes at the LHC, showing that enhanced sensitivity to the signal can be obtained with judicious selection of events. "

llm_abstract1 = "We present a comprehensive study of the production of massive photon pairs at hadron colliders within the framework of perturbative Quantum Chromodynamics (pQCD). The calculation is carried out in a fully differential manner, accounting for all relevant subprocesses and incorporating next-to-leading-order (NLO) corrections. The process under consideration involves the collision of hadrons, resulting in the creation of two photons with significant transverse momenta. Our analysis encompasses various kinematic regions, including both inclusive and exclusive regimes, enabling a detailed understanding of the underlying dynamics. Special attention is paid to the treatment of higher-order corrections, ensuring the reliability of our predictions across a wide range of energies and photon kinematics. Phenomenological implications of our results are discussed in the context of ongoing and future experimental efforts at hadron colliders, providing valuable insights into the physics of electromagnetic interactions in the strong coupling regime. Overall, our study constitutes a significant advancement in the theoretical description of photon pair production processes in high-energy hadronic collisions, with implications for precision phenomenology and the search for new physics phenomena."


# attack 1- splicing together outputs from different LLM responses
# attack 2= splicing together outputs from a human and an LLM response

#ratio is how much of the first text to include
def interweaveTexts(text1, text2, ratio):
    t1 = sent_tokenize(text1)
    t2 = sent_tokenize(text2)

    z = zip(t1, t2)
    res = [random.choices(val, weights=(ratio, abs(1 - ratio))) for val in z]
    final = " ".join(["".join([val for val in sublist]) for sublist in res])

    return final

#interweaveTexts(human_abstract1, llm_abstract1, 0.2)

# attack 3- replace words with the synonyms
#ration is how much of the original text to keep
def similarReplacement(text, ratio):
    sentence = []
    for word in word_tokenize(text):
        synonyms = []
        for syn in wordnet.synsets(word):
            for l in syn.lemmas():
                synonyms.append(l.name())
        if synonyms:
            rep = random.choice(tuple(set(synonyms)))
            choice = random.choices((word, rep), weights=(ratio, abs(1 - ratio)))[0]
            sentence.append(choice)
        else:
            sentence.append(word)

    reconstructedSentence = TreebankWordDetokenizer().detokenize(sentence)
    final = reconstructedSentence.replace("_", " ")
    return final

similarReplacement(human_abstract1, 0.8)