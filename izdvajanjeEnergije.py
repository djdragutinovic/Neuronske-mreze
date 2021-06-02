import os
import h5py
import eeglib
import numpy
import scipy
import scipy.io as sio
from scipy.signal import butter, filtfilt


if __name__ == '__main__':
    path = 'E:\\Djole\\EEG epohs 1'
    maliNiz = [];


# r=root, d=directories, f = files
    for r, d, f in os.walk(path):
        for file in f:
            if '01' in file:
                loadedFile = h5py.File(os.path.join(r, file), 'r')
                data = loadedFile.get('tempE')                              # data je 1x42 u slucaju pacijenta 1

                for i in range(0, data.size):                               # prolazim kroz svaki od 42 dela
                    data2 = data[i][0]                                      # data2 je 1x23

                    for j in range(0, loadedFile[data2].shape[0]):          # prolazim kroz svaki od 23 dela (channela)
                        data3 = loadedFile[data2][j][0]                     # data3 je 512x1800
                        jedanSat = []
                        numpyNiz = loadedFile[data3][()]

                        # prolazim kroz svaki od 1800 redova i radim bandPass 8 puta, potom transponujem,
                        # spajam u jedan segment od 2s i tako prosledjujem Welch metodi
                        for brojac in range(0, numpyNiz.shape[1]):
                            filteredData1 = eeglib.preprocessing.bandPassFilter(numpyNiz[:, brojac], sampleRate=256, highpass=0.5, lowpass=3)
                            filteredData2 = eeglib.preprocessing.bandPassFilter(numpyNiz[:, brojac], sampleRate=256, highpass=3, lowpass=6)
                            filteredData3 = eeglib.preprocessing.bandPassFilter(numpyNiz[:, brojac], sampleRate=256, highpass=6, lowpass=9)
                            filteredData4 = eeglib.preprocessing.bandPassFilter(numpyNiz[:, brojac], sampleRate=256, highpass=9, lowpass=12)
                            filteredData5 = eeglib.preprocessing.bandPassFilter(numpyNiz[:, brojac], sampleRate=256, highpass=12, lowpass=15)
                            filteredData6 = eeglib.preprocessing.bandPassFilter(numpyNiz[:, brojac], sampleRate=256, highpass=15, lowpass=18)
                            filteredData7 = eeglib.preprocessing.bandPassFilter(numpyNiz[:, brojac], sampleRate=256, highpass=18, lowpass=21)
                            filteredData8 = eeglib.preprocessing.bandPassFilter(numpyNiz[:, brojac], sampleRate=256, highpass=21, lowpass=25)

                            filteredData1 = numpy.transpose(filteredData1)
                            filteredData2 = numpy.transpose(filteredData2)
                            filteredData3 = numpy.transpose(filteredData3)
                            filteredData4 = numpy.transpose(filteredData4)
                            filteredData5 = numpy.transpose(filteredData5)
                            filteredData6 = numpy.transpose(filteredData6)
                            filteredData7 = numpy.transpose(filteredData7)
                            filteredData8 = numpy.transpose(filteredData8)

                            filteredDataFull = filteredData1
                            filteredDataFull = numpy.vstack([filteredDataFull, filteredData2])
                            filteredDataFull = numpy.vstack([filteredDataFull, filteredData3])
                            filteredDataFull = numpy.vstack([filteredDataFull, filteredData4])
                            filteredDataFull = numpy.vstack([filteredDataFull, filteredData5])
                            filteredDataFull = numpy.vstack([filteredDataFull, filteredData6])
                            filteredDataFull = numpy.vstack([filteredDataFull, filteredData7])
                            filteredDataFull = numpy.vstack([filteredDataFull, filteredData8])


                            filteredDataFull = numpy.asarray(filteredDataFull)

                            f, pxx = scipy.signal.welch(filteredDataFull, fs=256)
                            if len(jedanSat) == 0:                                  # ako je to prvi obradjeni segment
                                jedanSat = pxx
                            else:
                                jedanSat = numpy.vstack([jedanSat, pxx])            # ako nije, samo nadovezi


                        imeFajla = 'energija01' + '_' + str(i+1) + '_' + str(j+1) + '.mat'
                        sio.savemat(imeFajla, {'en': jedanSat})
                        print('Napravljen fajl ' + imeFajla)

