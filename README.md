# MailCruncher

This Python program cleans your inbox for you, by learning the criteria you use
to decide whether an unread email should be deleted or not (in fact sometimes you're interested in reading or keeping some of them). It uses a SGDClassifier from scikit-learn to do that.

Read [my blog post](http://gdelazzari.github.io/2017/06/04/emails-mess/) to know the story behind and to learn more details about the machine learning part.

## Running
To run this program you will need to install all the required dependencies. You can easly do that with the following command:

`sudo pip install -r requirements.txt`

You'll also need [this library](https://github.com/charlierguo/gmail) from [@charlierguo](https://github.com/charlierguo) (which is used to interface with the Gmail account).

Please note that this Python script will only run with Python 2 (because of that last library), so if your system default version is Python 3 you'll need to replace `pip` with `pip2` and run the script with the `python2` command.

## Issues
+ The code is a mess (it was written in about 7/8 hours)
+ Some parts are running very slow (like email deletion: spawning multiple threads could be a way to go)
+ It uses JSON files for >60MB databases (still I had no problems so far)
+ Not a lot of error checking
+ Sometimes emails are not cached correctly locally and it ends up fetching them again for no reason, which slows things down even more (this is something that requires a bit of debugging)
+ Saves your login details (password included) in plain text
+ Tested only on Linux
+ Only works with Gmail accounts
+ Sometimes weird things happen (not very often, but keep in mind that I've only tested the program with my email account and with the kind of emails my inbox was bloated with)
+ You may need to adjust some thresholds (like the "uncertainty threshold" or the "training vs. testing dataset split ratio")

## Pros
+ Cleans your inbox
+ It does that based on the criteria you taught it to use
+ You can clean thousands of unread emails just by teaching it a few hundreds samples
+ Puts emails it's uncertain about in a separate folder on your Gmail account
+ It's machine learning powered
+ It is quite accurate, it really surprised me
+ Training is fast
+ You can *absolutely* do a "safe" run before doing a real one, so you can preview how the classifier is taking its decisions

## Notes
+ Remember to train and test the classifier before doing any real run
+ Even if this is just an experiment, feedback is appreciated
+ Pull requests are welcome

## Disclaimer
I take no responsibility if you lose important emails. This software is provided
as it is, without any warranty. It's not a final product or an idiot-proof program
__by no means__, it's more of an experiment/toy program I created for fun. Use at your own risk. Although it works just on unread emails (which, because of the fact they're actually unread, probably means you don't care a lot about them), it has the power to trash your personal emails based on its decisions. You have been warned.
