# Music Generation using LSTM
I have downloaded the dataset from kaggle https://www.kaggle.com/datasets/soumikrakshit/classical-music-midi </br>
In this dataset there are MIDI files of classic piano music of various artists, I have used the Chopin's compositions.
Using music21 I have first parsed all the files and then using the function `extract_notes(file)` I have extracted the chords and notes out of the data to create a corpus. </br>
Then I have cleaned the data as there are some notes that have occured very rarely, so all those notes which have occured less than 100 times will be removed from the dataset.
After this I have created a dictionary to map the notes and their indices. Then I have encoded and split the corpus into small sequences of equal length of features and corresponding targets. Each of these features and target will contain the mapped index in the dictionary of the unique characters they signify. </br>
Then I have build the LSTM model adn train it on the proccessed data. To evaluate the model I have created a plot (Learning Curves). </br>
Then using this model I have generated some pieces of music in MIDI format.
