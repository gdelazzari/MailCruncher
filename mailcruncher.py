import os
import sys
import math
import json
import string

import getpass

import pickle

import gmail

import unicodedata

from random import randint

from bs4 import BeautifulSoup

from nltk import wordpunct_tokenize
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier


# Utility functions
# ------------------------------------------

def clearScreen():
    os.system('clear')

def getConsoleSize():
    rows, columns = os.popen('stty size', 'r').read().split()
    return int(rows), int(columns)

def writeJSON(data, filename):
    with open(filename, 'w') as outfile:
        json.dump(data, outfile)

def readJSON(filename):
    try:
        with open(filename) as data_file:
            data = json.load(data_file)
            return data
    except:
        return None

def fixUnicode(text):
    try:
        return unicodedata.normalize('NFKD', unicode(text, errors='ignore')).encode('ascii', 'ignore')
    except:
        try:
            return unicodedata.normalize('NFKD', text, errors='ignore').encode('ascii', 'ignore')
        except:
            return text

def cutText(text, maxlines, consolewidth):
    text = text.replace('\r', '')
    while '\n\n' in text:
        text = text.replace('\n\n', '\n')
    lines = text.split('\n')
    count = 0
    output = ""
    for line in lines:
        count += 1 + int(math.floor(len(line) / consolewidth))
        if count <= maxlines:
            output += line + '\n'
        else:
            break
    return output[:-1]

def clean_html(html):
    soup = BeautifulSoup(html, "html.parser")

    # kill all script and style elements
    for script in soup(["script", "style"]):
        script.extract()    # rip it out

    # get text
    text = soup.get_text()

    # break into lines and remove leading and trailing space on each
    lines = (line.strip() for line in text.splitlines())
    # break multi-headlines into a line each
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    # drop blank lines
    text = '\n'.join(chunk for chunk in chunks if chunk)

    return fixUnicode(text)


# Email utility functions
# ------------------------------------------
class GmailAccount:
    def __init__(self, login, password):
        self.g = gmail.login(login, password)
        self.dbname = login.split('@')[0] + ".json"
        self.db = readJSON(self.dbname)
        if self.db is None:
            self.db = {}
            self.writeDB()

    def writeDB(self):
        writeJSON(self.db, self.dbname)

    # Convert from a gmail.Message object to a Python dictionary
    def mailToDictionary(self, mail):
        result = {}
        if mail.fr is not None:
            result['fetched'] = True
            result['from'] = str(mail.fr)
            result['subject'] = fixUnicode(mail.subject)
            result['body'] = fixUnicode(mail.body)
            result['html'] = fixUnicode(mail.html)
        else:
            result['fetched'] = False
        return result

    # Convert from a Python dictionary to a gmail.Message object
    def dictionaryToMail(self, uid, d):
        mail = gmail.Message(self.g.inbox(), uid)
        if d['fetched'] == True:
            mail.fr = d['from']
            mail.subject = d['subject']
            mail.body = d['body']
            mail.html = d['html']
        return mail

    def insertIntoDB(self, emails, force=False):
        for mail in emails:
            if (not mail.uid in self.db) or (self.db[mail.uid]['fetched'] == False and mail.fr is not None) or (force == True):
                self.db[mail.uid] = self.mailToDictionary(mail)
        self.writeDB()

    def getUnreadInbox(self):
        if self.g.logged_in == False:
            return None
        unread = self.g.inbox().mail(unread=True)
        self.insertIntoDB(unread, force=False)
        return unread

    def fetchByUID(self, uid, update_db=True):
        if self.g.logged_in == False:
            return None
        if uid in self.db and self.db[uid]['fetched'] == True:
            return self.dictionaryToMail(uid, self.db[uid])
        mail = gmail.Message(self.g.inbox(), uid)
        try:
            mail.fetch()
            if update_db == True:
                self.insertIntoDB([mail], force=False)
            return mail
        except Exception as e:
            print("Can't update db:")
            print(e)
            return None

    def getMail(self, uid, auto_fetch=True):
        if self.g.logged_in == False:
            return None
        if not uid in self.db:
            return None
        if self.db[uid]['fetched'] == False:
            if auto_fetch == True:
                return self.fetchByUID(uid, update_db=True)
            else:
                return self.dictionaryToMail(uid, self.db[uid])
        else:
            return self.dictionaryToMail(uid, self.db[uid])


# Natural language processing
# ------------------------------------------

def detect_language(text):
    text = fixUnicode(text)
    languages_ratios = {}

    tokens = wordpunct_tokenize(text)
    words = [word.lower() for word in tokens]

    # Compute per language included in nltk number of unique stopwords appearing in analyzed text
    for language in stopwords.fileids():
        stopwords_set = set(stopwords.words(language))
        words_set = set(words)
        common_elements = words_set.intersection(stopwords_set)

        languages_ratios[language] = len(common_elements) # language "score"

    most_rated_language = max(languages_ratios, key=languages_ratios.get)
    if most_rated_language != 'italian' and most_rated_language != 'english':
        most_rated_language = 'english'
    return most_rated_language

def stemText(text, language):
    text = fixUnicode(text)
    punctuationMap = dict((ord(char), None) for char in string.punctuation)
    for asciival in punctuationMap:
        toremove = chr(asciival)
        text = text.replace(toremove, ' ')
    #text = fixUnicode(text)
    words = text.split()
    stemmer = SnowballStemmer(language)

    stemmedText = ""
    for word in words:
        if len(word) > 0:
            stemmedText += stemmer.stem(word) + " "

    return stemmedText


# Data functions
# ------------------------------------------

def fetchRandom(current_database, account, emails, number=5):
    def existsUID(elements, uid):
        for e in elements:
            if e.uid == uid:
                return True
        return False

    fetched = []
    while len(fetched) < number:
        try:
            selected = emails[randint(0, len(emails) - 1)]
            print("[{}/{}] Fetching email with uid #{}...\r".format(len(fetched), number, selected.uid)),
            sys.stdout.flush()
            if existsUID(fetched, selected.uid) == True: continue
            if selected.uid in current_database: continue
            selected = account.getMail(selected.uid, auto_fetch=True)
            if selected.fr is None: continue
            fetched.append(selected)
        except Exception as e:
            print("Error:", e)
            continue

    print("")

    return fetched

def askLabels(emails):
    labels = []

    rows, columns = getConsoleSize()

    count = 1
    for mail in emails:
        content = mail.body
        if content is None:
            content = clean_html(mail.html)
        clearScreen()
        print("[email " + str(count) + "/" + str(len(emails)) + "]")
        print("From: " + mail.fr)
        print("Subject: " + mail.subject)
        print("--------------------------------------------------------------------")
        print(cutText(content, rows - 7, columns))
        print("--------------------------------------------------------------------")
        label = raw_input("Would you keep an email like this? (y/n/quit): ")
        if label == 'y':
            labels.append(True)
        elif label == 'quit':
            break
        else:
            labels.append(False)
        count += 1

    return labels

def genMailBlob(mail):
    tags = ['<html>', '<p>', '<head>', '<body>', '<div>']
    blob = ""
    language = 'english'
    if mail.body is not None:
        if any(t in mail.body for t in tags):
            body_text = clean_html(mail.body)
            language = detect_language(body_text)
            blob += stemText(body_text, language)
        else:
            language = detect_language(mail.body)
            blob += stemText(mail.body, language)
    if mail.html is not None:
        html_text = clean_html(mail.html)
        blob += ' ' + stemText(html_text, detect_language(html_text))
    if mail.subject is not None:
        try:
            blob += ' ' + stemText(mail.subject, language)
        except:
            pass
    blob += ' ' + stemText(mail.fr, 'english')
    blob += ' ' + mail.fr.split('<')[-1].replace('>', '')
    return blob

def getBlobs(emails):
    blobs = []
    for mail in emails:
        blobs.append(genMailBlob(mail))
    return blobs

# Machine learning
# ------------------------------------------

def train(database, samples_percentage=0.85):
    data = []
    target = []

    samples_num = int(len(database) * samples_percentage)
    print("Using " + str(samples_num) + " items for training over " + str(len(database)) + " entries")

    for mailid in database:
        mail = database[mailid]
        data.append(mail['blob'])
        numerictarget = 0
        if mail['keep'] == True:
            numerictarget = 1
        target.append(numerictarget)

    clf = SGDClassifier(loss='modified_huber', penalty='l2', alpha=1e-3, n_iter=5, random_state=randint(0, 1000))
    cv = CountVectorizer(stop_words='english')
    tfidf = TfidfTransformer()
    wordCount = cv.fit_transform(data)
    wordFrequency = tfidf.fit_transform(wordCount)
    clf.fit(wordFrequency[0:samples_num], target[0:samples_num])

    print("\nTesting accuracy")
    print("----------------------------------------\n")
    right = 0
    avgprob = 0
    for i in range(len(database)):
        predicted = clf.predict(wordFrequency[i])
        probability = clf.predict_proba(wordFrequency[i])[0][predicted]
        if i == 0:
            avgprob = probability
        else:
            avgprob += probability
        if predicted == target[i]:
            right += 1
    avgprob /= len(database)
    print("Whole set: predicted " + str(right) + " right over " + str(len(database)))
    print(str((float(right) / float(len(database))) * 100.0) + "% accuracy")
    print("Average confidence: " + str(avgprob[0] * 100) + "%")

    right = 0
    total = len(database) - samples_num
    wrongdeletions = 0
    wrongkeeps = 0
    avgprob = 0
    for i in range(samples_num, len(database)):
        predicted = clf.predict(wordFrequency[i])
        probability = clf.predict_proba(wordFrequency[i])[0][predicted]
        if i == 0:
            avgprob = probability
        else:
            avgprob += probability
        #print("Predicted:", predicted, "Expected:", target[i])
        if (predicted != target[i]):
            if (predicted == 0):
                wrongdeletions += 1
            else:
                wrongkeeps += 1
        else:
            right += 1
    avgprob /= len(database) - samples_num
    print("")
    print("Testing set: predicted " + str(right) + " right over " + str(total))
    print(str(float(right) / float(total) * 100.0) + "% accuracy")
    print("Predicted " + str(wrongdeletions) + " wrong deletions")
    print("Predicted " + str(wrongkeeps) + " wrong keeps")
    print("Average confidence: " + str(avgprob[0] * 100) + "%")

    pickle.dump(clf, open("classifier.p", "wb"))
    pickle.dump(tfidf, open("tfidf.p", "wb"))
    pickle.dump(cv, open("cvectorizer.p", "wb"))

    print("\nThe classifier has been saved to disk\n")

def test(database):
    try:
        clf = pickle.load(open("classifier.p", "rb"))
        tfidf = pickle.load(open("tfidf.p", "rb"))
        cv = pickle.load(open("cvectorizer.p", "rb"))
    except:
        print("Error loading classifier")
        return

    data = []
    target = []

    for mailid in database:
        mail = database[mailid]
        data.append(mail['blob'])
        numerictarget = 0
        if mail['keep'] == True:
            numerictarget = 1
        target.append(numerictarget)

    wordCount = cv.transform(data)
    wordFrequency = tfidf.transform(wordCount)

    right = 0
    wrongdeletions = 0
    wrongkeeps = 0
    avgprob = 0
    for i in range(len(database)):
        predicted = clf.predict(wordFrequency[i])
        probability = clf.predict_proba(wordFrequency[i])[0][predicted]
        if i == 0:
            avgprob = probability
        else:
            avgprob += probability
        if (predicted != target[i]):
            if (predicted == 0):
                wrongdeletions += 1
            else:
                wrongkeeps += 1
        else:
            right += 1
    avgprob /= len(database)

    print("Self-test: predicted", right, "right over", len(database))
    print((float(right) / float(len(database))) * 100.0, "accuracy")
    print("Predicted", wrongdeletions, "wrong deletions")
    print("Predicted", wrongkeeps, "wrong keeps")
    print("Average confidence: " + str(avgprob * 100))


# Test and real runs
# ------------------------------------------

def analyzeUnread(account, unread, number=-1):
    if number == -1:
        number = len(unread)

    try:
        clf = pickle.load(open("classifier.p", "rb"))
        tfidf = pickle.load(open("tfidf.p", "rb"))
        cv = pickle.load(open("cvectorizer.p", "rb"))
    except:
        print("Error loading classifier")
        return

    data = []
    subjects = []
    froms = []
    uids = []

    print("Gathering data for " + str(number) + " emails...")

    count = 0
    discard = 0
    for u in unread:
        try:
            mail = account.getMail(u.uid, auto_fetch=True)
            if False:
                print mail.body
                print("")
                print mail
                print mail.fr
                print mail.subject
            blob = genMailBlob(mail)
            subject = fixUnicode(mail.subject)
            fr = str(mail.fr)
        except:
            discard += 1
            continue
        data.append(genMailBlob(mail))
        subjects.append(mail.subject)
        froms.append(fr)
        uids.append(u.uid)

        count += 1

        print("[{}%] Fetched and preprocessed {} out of {} ({} discarded)\r".format(int(float(count) / float(number) * 100), count, number, discard)),
        sys.stdout.flush()

        if count >= number:
            break

    wordCount = cv.transform(data)
    wordFrequency = tfidf.transform(wordCount)

    result = {}
    todelete = 0
    tokeep = 0
    uncertain = 0
    for i in range(0, count):
        predicted = clf.predict(wordFrequency[i])
        probabilities = clf.predict_proba(wordFrequency[i])[0]
        operation = ''
        if predicted == 1:
            probability = probabilities[1]
            tokeep += 1
            print("[ ]"),
            operation = 'keep'
        else:
            probability = probabilities[0]
            if probability > 0.7:
                todelete += 1
                print("[x]"),
                operation = 'delete'
            else:
                uncertain += 1
                print("[?]"),
                operation = 'uncertain'
        print(str(int(probability * 100)) + "%"),
        fr = froms[i].replace('\n', ' ').split(' ')[-1]
        while '  ' in fr:
            fr = fr.replace('  ', ' ')
        print(subjects[i][:40] + " | " + fr)
        result[uids[i]] = operation

    print("")
    print("Test run finished:")
    print(" > {} to keep".format(tokeep))
    print(" > {} to delete".format(todelete))
    print(" > {} uncertain".format(uncertain))
    print("")

    return result

def safeRun(account, unread, number=100):
    analyzeUnread(account, unread, number=number)

def realRun(account, unread):
    operations = analyzeUnread(account, unread, number=-1)
    print("!!! WARNING !!!")
    print("If you confirm, all the emails shown above and marked with [x]")
    print("will be deleted from your inbox. You will find them in the trash")
    print("folder anyway, but better to warn you of what's going to happen.")
    print("Type a full 'yes' if you agree.")
    print("")
    confirm = raw_input("Do you confirm? (yes/no): ")
    if confirm != 'yes':
        return
    applyLabels_str = raw_input("Label the uncertain emails as 'Uncertain'? (yes/no, default: yes): ")
    applyLabels = True
    if applyLabels_str == 'no' or applyLabels_str == 'n':
        applyLabels = False
    for uid in operations:
        mail = account.getMail(uid)
        operation = operations[uid]
        if operation == 'delete':
            print("Deleting #" + str(uid))
            mail.delete()
        elif operation == 'uncertain':
            mail.add_label("Uncertain")


# Main program
# ------------------------------------------

def printHeader():
    clearScreen()
    print("")
    print(" [MailCruncher] -- Machine Learning Powered ;)")
    print(" Giacomo De Lazzari, 2017")
    print(" Hire me when you have too many unread emails")
    print("")

def mainMenu(username, unread):
    printHeader()
    print(username + " currently has:")
    print(" > " + str(unread) + " unread emails")
    print("")
    print("This is the main menu, choose what to do")
    print("1. Input some training data")
    print("2. Train the classifier")
    print("3. Test the classifier")
    print("4. Perform a safe run (no emails deleted)")
    print("5. Perform a real run (BE CAREFUL)")
    print("q. Exit")
    print("")
    choice = raw_input("Type a choice: ")
    if choice == 'q':
        return -1
    else:
        try:
            code = int(choice)
            return code
        except:
            return -2

def firstRun():
    printHeader()
    print("You need to configure the Gmail account you want to use with the program")
    print("If you have two-factor authentication enabled, you must use a generated")
    print("password instead of your regular one (search for 'app passwords' on Google")
    print("to open the management panel).")
    print("")
    print("WARNING: Your password will be saved in plain text in account.json")
    print("         I take no responsibility for stolen password. If you want to fix")
    print("         this, the code is on GitHub, go on ;)")
    print("")
    username = raw_input("Full email address: ")
    password = getpass.getpass("Password: ")
    account = {
        'user': username,
        'pass': password
    }
    writeJSON(account, 'account.json')
    return account

def main():
    print("Reading your account login details...")
    login = readJSON('account.json')
    if login is None:
        login = firstRun()
        clearScreen()

    print("Connecting to your account...")
    try:
        account = GmailAccount(login['user'], login['pass'])
    except:
        print("\nThe credentials configured are not valid, couldn't connect to your mailbox.")
        print("Please delete the file 'account.json' and run the program again to reconfigure")
        print("your login details.\n")
        quit()

    print("Loading local database...")
    database = readJSON('db.json')
    if database is None:
        print("Database empty, initializing...")
        database = {}
        writeJSON(database, 'db.json')
    else:
        print("Successfully loaded " + str(len(database)) + " entries from the database")

    print("Querying for some unread messages...")
    unread = account.getUnreadInbox()

    while True:
        choice = mainMenu("giacky98.mail@gmail.com", len(unread))
        if choice == -1:
            break
        clearScreen()
        if choice == 1:
            print("Got " + str(len(unread)) + " unread messages to pick from")

            print("Fetching 25 random emails")
            fetched = fetchRandom(database, account, unread, 25)
            labels = askLabels(fetched)

            print("Generating email blobs...")
            blobs = getBlobs(fetched)

            print("Updating the database...")
            for i in range(len(labels)):
                mail = fetched[i]
                print(blobs[i])
                if not mail.uid in database:
                    database[mail.uid] = {
                        'blob': blobs[i],
                        'keep': labels[i]
                    }

            print("Writing database on disk...")
            writeJSON(database, 'db.json')
            raw_input("Press enter to continue")
        elif choice == 2:
            print("Training classifier...")
            train(database)
            raw_input("Press enter to continue")
        elif choice == 3:
            print("Testing classifier...")
            test(database)
            raw_input("Press enter to continue")
        elif choice == 4:
            print("Performing safe run...")
            safeRun(account, unread, number=-1)
            raw_input("Press enter to continue")
        elif choice == 5:
            print("Performing real run...")
            realRun(account, unread)
            raw_input("Press enter to continue")
            print("Requerying for unread messages...")
            unread = account.getUnreadInbox()


if __name__ == "__main__":
    main()
