import sys
from nltk.stem import WordNetLemmatizer

wordnet_lemmatizer = WordNetLemmatizer()


def main(filename):
    with open(
        "../data_aud_text/" + filename + ".text", mode="r", encoding="utf-8"
    ) as nonreg_trg_file:
        sentences = []
        # For Source, Target and FilePath
        for nonreg_trg_line in nonreg_trg_file:

            nonreg_trg_line = nonreg_trg_line.strip()
            src_wrds = nonreg_trg_line.split(" ")
            lemma = []
            for w in src_wrds:
                if wordnet_lemmatizer.lemmatize(w) == "wa":
                    lemma.append(w)
                elif wordnet_lemmatizer.lemmatize(w) == "ha":
                    lemma.append(w)
                else:
                    lemma.append(wordnet_lemmatizer.lemmatize(w))
            sentences.append(lemma)

            # print(lemma)
            # input()
    print("writing")
    with open("new_" + filename + ".text", "w") as train_text, open(
        "" + filename + ".words.txt", "w"
    ) as words_file:
        for line in sentences:
            train_text.write(" ".join(line) + "\n")
            for word in line:
                words_file.write(word + "\n")


if __name__ == "__main__":
    # print(sys.argv[1])
    main(sys.argv[1])
