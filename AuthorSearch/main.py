# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

from scholarly import scholarly
def print_hi(name):


    # Retrieve the author's data, fill-in, and print
    # Get an iterator for the author results
    search_query = scholarly.search_author('Aatif Bilal')
    # Retrieve the first result from the iterator
    first_author_result = next(search_query)
    #scholarly.pprint(first_author_result)
    for key, value in first_author_result.items():
        print(key, ' : ', value)
# Press the green button in the gutter to run the script.

if __name__ == '__main__':
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
