Methodo Shan

python main.py --method shan --image data\test\C081\1.png

Methodo Fergus

python main.py --method 

Metodo - Reti neurali end-to-end che sfruttano un'approccio supervisionato (o self-supervised), e con la classica dinamica train set - test set. 
Starei lontano dagli approcci DIP e da quelli Plug-and-Play.

--- Esperimento #1 - Sensibilita' lambda_prior

Test A.1 (impatto kernel_size):

# Esegui con un kernel di motion blur da 35x35 e prova a stimarlo con 35
python main.py --image data/text.png --synthetic --blur_type motion --kernel_size 35 --motion_len 40

# Ora prova a stimarlo con una dimensione sbagliata (più piccola)
python main.py --image data/text.png --synthetic --blur_type motion --kernel_size 21 --motion_len 40

(Nota: in questo test, il blur generato sarà sempre 35x35, ma l'algoritmo userà 21x21 per la stima, permettendoti di vedere l'impatto dell'errore)

Test A.2 (sensibilità a lambda):

# Valore basso di lambda_prior
python main.py --image data/landscape_sharp.png --synthetic --kernel_size 41 --lambda_prior 0.001

# Valore alto di lambda_prior
python main.py --image data/landscape_sharp.png --synthetic --kernel_size 41 --lambda_prior 0.01

Test A.3 (blur gaussiano):

# Genera e risolvi un blur gaussiano
python main.py --image data/checkerboard.png --synthetic --blur_type gaussian --kernel_size 31 --gaussian_sigma 4

-- Esperimento #2 - L'Importanza della dimensione del kernel

Rispondere alla domanda: "Cosa succede se dico all'algoritmo di cercare un blur di 20 pixel, ma in realta' è di 40? O se gli do troppo spazio?"

# Test 2.1: Kernel STIMATO troppo piccolo (25x25) per un blur REALE che ci starebbe a malapena
# L'algoritmo non ha abbastanza spazio per "disegnare" il kernel che trova.
python main.py --image data/test/C081/15.png --synthetic --motion_len 40 --kernel_size 25 --no_show

# Test 2.2: Kernel STIMATO della dimensione giusta (45x45)
# Diamo all'algoritmo lo spazio corretto per lavorare.
python main.py --image data/test/C081/15.png --synthetic --motion_len 40 --kernel_size 45 --no_show

# Test 2.3: Kernel STIMATO troppo grande (65x65)
# Diamo all'algoritmo fin troppo spazio. Potrebbe confondersi e aggiungere rumore.
python main.py --image data/test/C081/15.png --synthetic --motion_len 40 --kernel_size 65 --no_show
