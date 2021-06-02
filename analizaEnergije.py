import numpy
import scipy.io as sio
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import confusion_matrix


if __name__ == '__main__':

    podaci = []             # za cuvanje podataka koji ce biti prosledjeni modelu na obucavanje
    izlazi = []             # 0 ili 1
    podaciTest = []
    izlaziTest = []
    skupChannela = []
    W = 3                   # koliko uzastopnih delova se spaja
    S = 10                  # u stvari 20 (10 delova po 2 sekunde), koliko prvih sekundi od epi napada uzimam za analizu

    # nizZaUcitavanjeNonSeizurePodataka = [1,2,5,6,7,8,9,10,11,12,13,14,17,19,20,22,23,24,25,28,29,30,31,32]
    # nizZaUcitavanjeNonSeizurePodataka = [2,5,8,11,14,17,19,22,25]
    nizZaUcitavanjeNonSeizurePodataka = [5, 17]

    for i in nizZaUcitavanjeNonSeizurePodataka:             # prolazim kroz svaki fajl koji sam definisao za ucitavanje
        skupChannela = []
        for j in range(1, 24):
            imeFajla = 'Pacijenti\\energija01_' + str(i) + '_' + str(j) + '.mat'           # za analizu koristim prvog pacijenta
            loadedFile = sio.loadmat(imeFajla)
            data = loadedFile.get('en')
            skupChannela.append(data)                       # sacuvam sve energije channel-a jednog sata, da ih ne bih ucitavao non-stop
        skupChannela = numpy.asarray(skupChannela)

        brojSegmenata = int(data.shape[0]/8 - W + 1)                # koliko cu segmenata od 6 sekundi imati u skupu (14400/8-3+1 = 1798)
        for indeksSekunde in range(0, brojSegmenata):               # brojac za pravljenje delova od 6 sekundi
            sestSekundi = []
            for w in range(0, W):                                   # posto spajam 3 dela, ucitacu 3 puta i samo nadovezivati
                for brojac in range(0, skupChannela.shape[0]):      # prolazim kroz svaki od 23 kanala
                    pocetakPreuzimanja = 8 * indeksSekunde + w*8    # 2 sekunde su 8 redova, pa definisem pocetak i kraj preuzimanja u zavisnosti od w
                    krajPreuzimanja = 8 * (indeksSekunde+1) + w*8
                    channel = skupChannela[brojac]
                    dveSekundeJednogChannela = channel[pocetakPreuzimanja:krajPreuzimanja, :]
                    if len(sestSekundi) == 0:
                        sestSekundi = dveSekundeJednogChannela
                    else:
                        sestSekundi = numpy.vstack([sestSekundi, dveSekundeJednogChannela])
            sestSekundi = numpy.asarray(sestSekundi)
            podaci.append(sestSekundi)
            izlazi.append(0)

# ----------------------------

# U narednom delu prolazim kroz fajlove koji imaju epi napad i pristupam tacno tom delu gde je napad

    skupChannela = []
    for j in range(1, 24):
        imeFajla = 'Pacijenti\\energija01_3_' + str(j) + '.mat'
        loadedFile = sio.loadmat(imeFajla)
        data = loadedFile.get('en')
        skupChannela.append(data)
    skupChannela = numpy.asarray(skupChannela)

    pocetakEpiNapadaZa03 = int(2996/2)
    brojSegmenataEpiNapada = S-W+1
    for indeksSekunde in range(pocetakEpiNapadaZa03, pocetakEpiNapadaZa03 + brojSegmenataEpiNapada):
        sestSekundiNapada = []
        for w in range(0, W):
            for brojac in range(0, skupChannela.shape[0]):
                pocetakPreuzimanja = 8 * indeksSekunde + w*8
                krajPreuzimanja = 8 * (indeksSekunde+1) + w*8
                channel = skupChannela[brojac]
                dveSekundeEpiNapada = channel[pocetakPreuzimanja:krajPreuzimanja, :]        # preuzimam tacno te redove koji mi trebaju
                if len(sestSekundiNapada) == 0:
                    sestSekundiNapada = dveSekundeEpiNapada
                else:
                    sestSekundiNapada = numpy.vstack([sestSekundiNapada, dveSekundeEpiNapada])
        sestSekundiNapada = numpy.asarray(sestSekundiNapada)
        podaci.append(sestSekundiNapada)
        izlazi.append(1)

# ---------------------

    skupChannela = []
    for j in range(1, 24):
        imeFajla = 'Pacijenti\\energija01_15_' + str(j) + '.mat'
        loadedFile = sio.loadmat(imeFajla)
        data = loadedFile.get('en')
        skupChannela.append(data)
    skupChannela = numpy.asarray(skupChannela)

    pocetakEpiNapadaZa15 = int(1732 / 2)
    brojSegmenataEpiNapada = S - W + 1
    for indeksSekunde in range(pocetakEpiNapadaZa15, pocetakEpiNapadaZa15 + brojSegmenataEpiNapada):
        sestSekundiNapada = []
        for w in range(0, W):
            for brojac in range(0, skupChannela.shape[0]):
                pocetakPreuzimanja = 8 * indeksSekunde + w * 8
                krajPreuzimanja = 8 * (indeksSekunde + 1) + w * 8
                channel = skupChannela[brojac]
                dveSekundeEpiNapada = channel[pocetakPreuzimanja:krajPreuzimanja, :]
                if len(sestSekundiNapada) == 0:
                    sestSekundiNapada = dveSekundeEpiNapada
                else:
                    sestSekundiNapada = numpy.vstack([sestSekundiNapada, dveSekundeEpiNapada])
        sestSekundiNapada = numpy.asarray(sestSekundiNapada)
        podaci.append(sestSekundiNapada)
        izlazi.append(1)
# --------------------

    skupChannela = []
    for j in range(1, 24):
        imeFajla = 'Pacijenti\\energija01_18_' + str(j) + '.mat'
        loadedFile = sio.loadmat(imeFajla)
        data = loadedFile.get('en')
        skupChannela.append(data)
    skupChannela = numpy.asarray(skupChannela)

    pocetakEpiNapadaZa18 = int(1720 / 2)
    brojSegmenataEpiNapada = S - W + 1
    for indeksSekunde in range(pocetakEpiNapadaZa18, pocetakEpiNapadaZa18 + brojSegmenataEpiNapada):
        sestSekundiNapada = []
        for w in range(0, W):
            for brojac in range(0, skupChannela.shape[0]):
                pocetakPreuzimanja = 8 * indeksSekunde + w * 8
                krajPreuzimanja = 8 * (indeksSekunde + 1) + w * 8
                channel = skupChannela[brojac]
                dveSekundeEpiNapada = channel[pocetakPreuzimanja:krajPreuzimanja, :]
                if len(sestSekundiNapada) == 0:
                    sestSekundiNapada = dveSekundeEpiNapada
                else:
                    sestSekundiNapada = numpy.vstack([sestSekundiNapada, dveSekundeEpiNapada])
        sestSekundiNapada = numpy.asarray(sestSekundiNapada)
        podaci.append(sestSekundiNapada)
        izlazi.append(1)

# ------------------------------------------------------------------------------------------------------------------

# Ovde ucitavam za potrebe testiranja


    skupChannela = []
    for j in range(1, 24):
        imeFajla = 'Pacijenti\\energija01_42_' + str(j) + '.mat'
        loadedFile = sio.loadmat(imeFajla)
        data = loadedFile.get('en')
        skupChannela.append(data)
    skupChannela = numpy.asarray(skupChannela)

    # counter = 0
    brojSegmenata = int(data.shape[0] / 8 - W + 1)
    for indeksSekunde in range(0, brojSegmenata, 6):        # za potrebe testiranja necu koristiti ceo skup od sat vremena, nego cu uzeti svaki 6. segment
        sestSekundi = []
        for w in range(0, W):
            for brojac in range(0, skupChannela.shape[0]):
                pocetakPreuzimanja = 8 * indeksSekunde + w * 8
                krajPreuzimanja = 8 * (indeksSekunde + 1) + w * 8
                channel = skupChannela[brojac]
                dveSekundeJednogChannela = channel[pocetakPreuzimanja:krajPreuzimanja, :]
                if len(sestSekundi) == 0:
                    sestSekundi = dveSekundeJednogChannela
                else:
                    sestSekundi = numpy.vstack([sestSekundi, dveSekundeJednogChannela])
        sestSekundi = numpy.asarray(sestSekundi)
        podaciTest.append(sestSekundi)
        izlaziTest.append(0)




    skupChannela = []
    for j in range(1, 24):
        imeFajla = 'Pacijenti\\energija01_16_' + str(j) + '.mat'
        loadedFile = sio.loadmat(imeFajla)
        data = loadedFile.get('en')
        skupChannela.append(data)
    skupChannela = numpy.asarray(skupChannela)

    pocetakEpiNapadaZa16 = int(1016 / 2)
    brojSegmenataEpiNapada = S - W + 1

    for indeksSekunde in range(pocetakEpiNapadaZa16, pocetakEpiNapadaZa16 + brojSegmenataEpiNapada):
        sestSekundiNapada = []
        for w in range(0, W):
            for brojac in range(0, skupChannela.shape[0]):
                pocetakPreuzimanja = 8 * indeksSekunde + w * 8
                krajPreuzimanja = 8 * (indeksSekunde + 1) + w * 8
                channel = skupChannela[brojac]
                dveSekundeEpiNapada = channel[pocetakPreuzimanja:krajPreuzimanja, :]
                if len(sestSekundiNapada) == 0:
                    sestSekundiNapada = dveSekundeEpiNapada
                else:
                    sestSekundiNapada = numpy.vstack([sestSekundiNapada, dveSekundeEpiNapada])
        sestSekundiNapada = numpy.asarray(sestSekundiNapada)
        podaciTest.append(sestSekundiNapada)
        izlaziTest.append(1)

# ------------------------------------------------------------------------------------------------------------------

    podaci = numpy.array(podaci)
    izlazi = numpy.array(izlazi)
    podaciTest = numpy.array(podaciTest)
    izlaziTest = numpy.array(izlaziTest)

    izlazi = izlazi.transpose()
    izlaziTest = izlaziTest.transpose()

    podaci = podaci.reshape(podaci.shape[0], (podaci.shape[1]*podaci.shape[2]))                         # radim reshape da bih dobio 2D niz koji se moze proslediti za analizu
    podaciTest = podaciTest.reshape(podaciTest.shape[0], (podaciTest.shape[1]*podaciTest.shape[2]))
    scaling = MinMaxScaler(feature_range=(-1, 1))
    scaling.fit(podaci)
    podaciSc = scaling.transform(podaci)
    podaciTestSc = scaling.transform(podaciTest)

# ----------------------------------------------------------------


    # kreiranje troslojne mreze
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(71208,)),
        keras.layers.Dense(16, activation=tf.nn.relu),
        keras.layers.Dense(16, activation=tf.nn.relu),
        keras.layers.Dense(1, activation=tf.nn.sigmoid),
    ])

    adam = tf.keras.optimizers.Adam(learning_rate=0.003)                    # kreiranje optimizera ()

    model.compile(optimizer=adam,                                           # "kompajliranje" neuronske mreze
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    history = model.fit(podaciSc, izlazi, epochs=15)
    test_loss, test_acc = model.evaluate(podaciTestSc, izlaziTest)
    print("Accuracy score of Sequential Neural Network: ", round(100 * test_acc, 2), "%")



    # rnn
    model = keras.Sequential()
    podaciSc = numpy.expand_dims(podaciSc, 1)
    podaciTestSc = numpy.expand_dims(podaciTestSc, 1)

    # Recurrent layer
    model.add(keras.layers.GRU(16))
    # Output layer
    model.add(keras.layers.Dense(1, activation=tf.nn.tanh))

    adam = tf.keras.optimizers.Adam(learning_rate=0.001)
    # Compile the model
    model.compile(
        optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])

    history = model.fit(podaciSc, izlazi, epochs=15)
    test_loss, test_acc = model.evaluate(podaciTestSc, izlaziTest)
    print("Accuracy score of Recurrent Neural Network: ", round(100 * test_acc, 2), "%")
