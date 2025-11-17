from sklearn.datasets import fetch_20newsgroups
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer

# 1. Print emails.target_names to see the different categories.
emails = fetch_20newsgroups()
print("Available email categories:", emails.target_names)

# 2, 5. Select specific categories and create the training set.
# 14. Change these categories later to test different datasets.
categories_list = ['rec.sport.baseball', 'rec.sport.hockey']
# categories_list = ['comp.sys.ibm.pc.hardware','rec.sport.hockey'] # Use for Task 14

train_emails = fetch_20newsgroups(
    categories=categories_list,
    subset='train',
    shuffle=True,
    random_state=108
)

# 6. Create the test set.
test_emails = fetch_20newsgroups(
    categories=categories_list,
    subset='test',
    shuffle=True,
    random_state=108
)

# 3. Print the email at index 5 in the training data.
print("\nEmail at index 5:\n", train_emails.data[5])

# 4. Print the label of the email at index 5.
print("\nLabel of email at index 5:", train_emails.target[5])
print("Corresponding category name:", train_emails.target_names[train_emails.target[5]])

# 7. Create a CountVectorizer object.
counter = CountVectorizer()

# 8. Tell counter what possible words can exist in the emails.
# Fitting on the combined data ensures all possible words are learned.
counter.fit(test_emails.data + train_emails.data)

# 9. Make a list of the counts of words in the training set.
train_counts = counter.transform(train_emails.data)

# 10. Make a list of the counts of words in the test set.
test_counts = counter.transform(test_emails.data)

# 11. Create a MultinomialNB object named classifier.
classifier = MultinomialNB()

# 12. Train the classifier.
classifier.fit(train_counts, train_emails.target)

# 13. Test the classifier by printing its accuracy score.
accuracy = classifier.score(test_counts, test_emails.target)
print(f"\nClassifier accuracy: {accuracy:.4f}")

# 14, 15. The accuracy score measures the percentage of correct classifications.
# Try changing the categories_list variable at the top of the script and running the code again.
