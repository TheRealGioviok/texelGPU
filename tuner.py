
# The features are already extracted, and are stored in the file called "features.txt".

# In order to maximize the speed of the tuning, we will use the following tricks:
# 1. Since Texel's tuning method calculates a sigmoid on the average error, we will use numpy to calculate all the scores and errors at once. Note that the tuning process does not require to play the games, as we have already extracted ~11.5 million positions from training games.
# 2. We will extract the features of the board only once, and then use the features to calculate the scores and errors.
# 3. For every weight we modify, we will keep track of the "momentum" of the weight, and use it to update the weight.

# First, we must import the necessary libraries
import cupy as np  # To calculate the sigmoid and the average error
import random       # To generate random numbers
import os           # To check if the user wants to stop the tuning
import numpy



# The evaluate function will be given a batch of positions, and will return a batch of scores
def evaluate(positions: np.array, phases: np.array, weights: np.array) -> np.array:
    # Now we can calculate the score
    mgScore = np.sum(positions * weights[:317], axis = 1)
    egScore = np.sum(positions * weights[317:(317 * 2)], axis = 1)
    # taper the scores based on game phase. In particular, gamephase ranges from 0 to 24, where 0 is the opening, and 24 is the endgame
    # We will linearly interpolate between the opening and endgame scores
    tapered = (mgScore * phases + egScore * (24 - phases)) / 24
    return tapered

# Since scores are in the range [-N, N], we need to convert them to the range [0, 1]
def convert_scores(scores: np.array, K: float) -> np.array:
    # While the evaluation function returns scores in the range [-N, N], the targets are in the range [0, 1], so we need to convert them
    # We use a sigmoid function defined as follows:
    # L = (1 / (1 + e^(-K * E)))
    # Where E is the evaluation score, and K is a constant, that we will calculate at the start of the training
    # In particular, K is chosen such as it minimized the starting error
    convertedScores = 1 / (1 + np.exp(-K * scores))
    
    return convertedScores

# The loss function is the one described in the paper
def loss(normScores: np.array, targets: np.array) -> float:
    # The loss function is L2 loss. We may consider to use normalizations, but for now, we'll just use the L2 loss
    return np.mean(np.square(normScores - targets))

def saveWeights(weights: np.array, filename: str, binary: bool = False):
    if binary:
        # Save the weights in binary format
        with open(filename, "wb") as f:
            weights.tofile(f)
        return
    # move weights array to cpu
    ws = list( np.asnumpy(weights) )
    # We save the ws so that we can immediately use them in the engine.
    # Create the file if it doesn't exist
    with open(filename, "w") as f:
        # First 10 ws are material ws
        f.write("constexpr Score materialMg[6] = {")
        for i in range(5):
            f.write(str(round(ws[i])) + ", ")
        f.write(" 0};\n")
        f.write("constexpr Score materialEg[6] = {")
        for i in range(317, 5 + 317):
            f.write(str(round(ws[i])) + ", ")
        f.write(" 0};\n")
        # Then we have pawn psqt
        f.write("constexpr Score pawnPsqtMg[64] = {")
        base = 6
        for i in range(64):
            f.write(str(round(ws[base + i])) + ", ")
            if i % 8 == 7:
                f.write("\n")
        f.write("};\n")
        base += 64
        pieces = ["knight", "bishop", "rook", "queen", "king"]
        for piece in pieces:
            # 32 ws for pieces psqt, but we mirror them
            f.write("constexpr Score %sPsqtMg[64] = {" % piece)
            for i in range(64):
                sqx = min(i % 8, 7 - i % 8)
                sqy = i // 8
                f.write(str(round(ws[base + sqy * 4 + sqx])) + ", ")
                if i % 8 == 7:
                    f.write("\n")
            f.write("};\n")
            base += 32

        # Now eg psqt
        # base is already set
        # first pawns
        base = 317 + 6
        f.write("constexpr Score pawnPsqtEg[64] = {")
        for i in range(64):
            f.write(str(round(ws[base + i])) + ", ")
            if i % 8 == 7:
                f.write("\n")
        f.write("};\n")
        base += 64
        for piece in pieces:
            # 32 ws for pieces psqt, but we mirror them
            f.write("constexpr Score %sPsqtEg[64] = {" % piece)
            for i in range(64):
                sqx = min(i % 8, 7 - i % 8)
                sqy = i // 8
                f.write(str(round(ws[base + sqy * 4 + sqx])) + ", ")
                if i % 8 == 7:
                    f.write("\n")
            f.write("};\n")
            base += 32
        mobbase = 6 + 64 + 32 * 5
        base = mobbase
        # Mobility lookups (size 9, 14, 15, 28)
        f.write("constexpr Score mobilityKnight[9] = {")
        for i in range(9):
            f.write(str(round(ws[base + i])) + ", ")
        f.write("};\n")
        base += 9
        f.write("constexpr Score mobilityBishop[14] = {")
        for i in range(14):
            f.write(str(round(ws[base + i])) + ", ")
        f.write("};\n")
        base += 14
        f.write("constexpr Score mobilityRook[15] = {")
        for i in range(15):
            f.write(str(round(ws[base + i])) + ", ")
        f.write("};\n")
        base += 15
        f.write("constexpr Score mobilityQueen[28] = {")
        for i in range(28):
            f.write(str(round(ws[base + i])) + ", ")
        f.write("};\n")
        base += 28
        base = mobbase + 317
        # Eg mobility lookups (size 9, 14, 15, 28)
        f.write("constexpr Score mobilityKnightEg[9] = {")
        for i in range(9):
            f.write(str(round(ws[base + i])) + ", ")
        f.write("};\n")
        base += 9
        f.write("constexpr Score mobilityBishopEg[14] = {")
        for i in range(14):
            f.write(str(round(ws[base + i])) + ", ")
        f.write("};\n")
        base += 14
        f.write("constexpr Score mobilityRookEg[15] = {")
        for i in range(15):
            f.write(str(round(ws[base + i])) + ", ")
        f.write("};\n")
        base += 15
        f.write("constexpr Score mobilityQueenEg[28] = {")
        for i in range(28):
            f.write(str(round(ws[base + i])) + ", ")
        f.write("};\n")
        base = mobbase + 9 + 14 + 15 + 28
        # weak doubled pawns
        f.write("constexpr Score pawnWeakDoubledMg = %f;\n" % round(ws[base]))
        f.write("constexpr Score pawnWeakDoubledEg = %f;\n" %
                round(ws[base + 317]))
        base += 1
        # doubled pawns
        f.write("constexpr Score pawnDoubledMg = %f;\n" % round(ws[base]))
        f.write("constexpr Score pawnDoubledEg = %f;\n" %
                round(ws[base + 317]))
        base += 1
        # backward pawns
        f.write("constexpr Score pawnBackwardMg = %f;\n" % round(ws[base]))
        f.write("constexpr Score pawnBackwardEg = %f;\n" %
                round(ws[base + 317]))
        base += 1
        # isolated pawns
        f.write("constexpr Score pawnIsolatedMg = %f;\n" % round(ws[base]))
        f.write("constexpr Score pawnIsolatedEg = %f;\n" %
                round(ws[base + 317]))
        base += 1
        # passed bonus lookup table [297 : 306]
        f.write("constexpr Score passedBonusMg[9] = {")
        for i in range(9):
            f.write(str(round(ws[base + i])) + ", ")
        f.write("};\n")
        base += 317
        f.write("constexpr Score passedBonusEg[9] = {")
        for i in range(9):
            f.write(str(round(ws[base + i])) + ", ")
        f.write("};\n")
        base -= 317
        base += 9
        # safe pawns bonus
        f.write("constexpr Score pawnSafeMg = %f;\n" % round(ws[base]))
        f.write("constexpr Score pawnSafeEg = %f;\n" % round(ws[base + 317]))
        base += 1
        # bishop pair bonus
        f.write("constexpr Score bishopPairMg = %f;\n" % round(ws[base]))
        f.write("constexpr Score bishopPairEg = %f;\n" % round(ws[base + 317]))
        base += 1
        # minor pieces in attack span
        f.write("constexpr Score minorInAttackSpanMg = %f;\n" %
                round(ws[base]))
        f.write("constexpr Score minorInAttackSpanEg = %f;\n" %
                round(ws[base + 317]))
        base += 1
        # rooks in attack span
        f.write("constexpr Score rookInAttackSpanMg = %f;\n" % round(ws[base]))
        f.write("constexpr Score rookInAttackSpanEg = %f;\n" %
                round(ws[base + 317]))
        base += 1
        # queen in attack span
        f.write("constexpr Score queenInAttackSpanMg = %f;\n" %
                round(ws[base]))
        f.write("constexpr Score queenInAttackSpanEg = %f;\n" %
                round(ws[base + 317]))
        base += 1
        # nonlinear pawn and knights closed position bonus
        f.write("constexpr Score knightPawnsMg = %f;\n" % round(ws[base]))
        f.write("constexpr Score knightPawnsEg = %f;\n" %
                round(ws[base + 317]))
        base += 1

def shuffle(x: np.array, y: np.array, z: np.array):
    # Shuffle two arrays in the same way
    assert len(x) == len(y) == len(z)
    # Use the Fisher-Yates shuffle
    for i in range(len(x) - 1, 0, -1):
        j = random.randint(0, i)
        x[i], x[j] = x[j], x[i]
        y[i], y[j] = y[j], y[i]
        z[i], z[j] = z[j], z[i]
    return x, y, z

def clearScreen():
    # Clear the screen
    if os.name == "nt":
        os.system("cls")
    else:
        os.system("clear")

def texel(weights: np.array, positions: np.array, targets: np.array, phases: np.array, K: float, lr: float = 20, EPOCHS: int = 10000):
    # Texel tuning is easy to implement. We just variate each feature by a margin and see which one is better. At the end, we variate based on the total variation.
    # Note that the learning rate is different from the learning rate used in the other tuning methods. It represent the sum of the variation of all features.
    datasetlen = len(positions)
    # calculate the initial loss 
    initialLoss = loss(convert_scores(evaluate(positions[15], phases[15], weights), K), targets[15])
    currLoss = initialLoss
    valLoss = "none"
    for epoch in range(EPOCHS):
        if epoch % 16 == 15:
            lr = lr * 0.9
        for b in range(15):
            # Get the batch (the positions are already subdivided in batches)
            batchsize = len(positions[b])
            # Now we can variate each feature
            wcpy = weights.copy()
            print()
            clearScreen()
            print("====== GPU Texel tuning ======")
            print("Dataset length: " + str(datasetlen))
            print("Initial loss: " + str(initialLoss))
            print("K: " + str(K))
            print("Epoch: " + str(epoch))
            print("Current loss: " + str(currLoss))
            print("Learning rate: " + str(lr))
            print("Batch: " + str(b) + "    ")
            print("Batch size: " + str(batchsize))
            print("Validation loss: " + str(valLoss))
            someChange = False
            # randomize the order of the features processed
            order = list(range(len(weights)))
            random.shuffle(order)
            cnt = 0
            currLoss = loss(convert_scores(evaluate(positions[b], phases[b], weights), K), targets[b])
            for w in order:
                if (w >= 6 and w <= 6 + 7) or (w >= 317 + 6 and w <= 317 + 6 + 7) or (w >= 6 + 56 and w <= 6 + 56 + 7) or (w >= 317 + 6 + 56 and w <= 317 + 6 + 56 + 7): #skip pawns on 1st and 8th rank
                    continue
                cnt += 1
                print("\rVariating feature " + str(w) + "(" + str(cnt) + ")" + " " * 5, end="")

                bestChange = 0 # The best change we have made
                bestLoss = currLoss # The best loss we have made
                # Positive change:
                weights[w] += lr # avoiding floating point errors
                newLoss = loss(convert_scores(evaluate(positions[b], phases[b], weights), K), targets[b])
                if newLoss < bestLoss:
                    # This change was good
                    bestLoss = newLoss
                    bestChange = lr
                # Negative change:
                weights[w] = wcpy[w] - lr # avoiding floating point errors
                newLoss = loss(convert_scores(evaluate(positions[b], phases[b], weights), K), targets[b])
                if newLoss < bestLoss:
                    # This change was good
                    bestLoss = newLoss
                    bestChange = -lr
                # Restore the weights
                weights[w] = wcpy[w] + bestChange
            
            # Save the weights
            saveWeights(weights, "weights.txt")
            # Export the weights as binary dump of the weights
            saveWeights(weights, "weights.bin", binary = True)

        # Calculate the validation loss on the last batch
        valLoss = loss(convert_scores(evaluate(positions[15], phases[15], weights), K), targets[15])
        # save the weights after each epoch
        saveWeights(weights, "weightsvalidated.txt")
        # Export the weights as binary dump of the weights
        saveWeights(weights, "weightsvalidated.bin", binary = True)
    return weights

def loadDataset(filename:str = "features.bin"):
    # The dataset contains already converted features. The file is in binary format.
    # Each byte represents a feature. The each item is 319 bytes long.
    # The first 317 bytes are the features, 318 is gamephase, the last byte is the target.
    x,y,z = [],[],[]
    with open(filename, "rb") as f:
        while True:
            # read the features file 256 * 319 bytes at a time
            read = f.read(319)
            # if we reached the end of the file, break
            if len(read) == 0 or len(x) >= 4194304: #do not use all the dataset until we figure out why the results are so bad
                break
            
            # convert the features to a numpy array
            features = numpy.frombuffer(read, dtype=np.int8)
            # The first 317 features are normal features
            x.append(features[:317])
            # The 318th feature is the gamephase, which we store separately (z)
            z.append(features[317])
            # The last feature is the target
            y.append(features[318])
            
            # print the progress
            if len(x) & 0x3FF == 0:
                print("\rLoaded %s lines" % len(x), end = "")
    print("\rLoaded %s lines" % len(x))
    # Divide the target values by 2 (normalization) and convert to pure numpy arrays
    x = numpy.array(x, dtype = np.float32)
    y = numpy.array(y, dtype = np.float32) / 2
    z = numpy.array(z, dtype = np.float32)


    print(x.shape, y.shape, z.shape)
    # Shuffle the dataset
    x, y, z = shuffle(x, y, z)
    print(x.shape, y.shape, z.shape)

    # divide into 16 batches (originak size is (nsamples, 317) for x and (nsamples,) for y)
    nsamples = x.shape[0]
    x = x.reshape((16, nsamples // 16, 317))
    y = y.reshape((16, nsamples // 16))
    z = z.reshape((16, nsamples // 16))

    print(x.shape, y.shape, z.shape)
    print(y)
    # Move to the GPU (np is cupy, so this is a GPU operation)
    x = np.array(x)
    y = np.array(y)
    z = np.array(z)
    print("Dataset loaded to GPU")
    # return the dataset
    return x, y, z
        
def defaultWeights():
    # Perseus default weights
    weights = np.zeros(317 * 2)
    weights[0] = 103
    weights[0 + 317] = 150
    weights[1] = 559
    weights[1 + 317] = 567
    weights[2] = 595
    weights[2 + 317] = 606
    weights[3] = 876
    weights[3 + 317] = 946
    weights[4] = 1782
    weights[4 + 317] = 1809
    weights[5] = 0
    weights[5 + 317] = 0
    mgpwn = [0,   0,   0,   0,   0,   0,  0,   0,
             98, 134,  61,  95,  68, 126, 34, -11,
             -6,   7,  26,  31,  65,  56, 25, -20,
             -14,  13,   6,  21,  23,  12, 17, -23,
             -27,  -2,  -5,  12,  17,   6, 10, -25,
             -26,  -4,  -4, -10,   3,   3, 33, -12,
             -35,  -1, -20, -23, -15,  24, 38, -22,
             0,   0,   0,   0,   0,   0,  0,   0]
    egpwn = [0,   0,   0,   0,   0,   0,   0,   0,
             178, 173, 158, 134, 147, 132, 165, 187,
             94, 100,  85,  67,  56,  53,  82,  84,
             32,  24,  13,   5,  -2,   4,  17,  17,
             13,   9,  -3,  -7,  -7,  -8,   3,  -1,
             4,   7,  -6,   1,   0,  -5,  -1,  -8,
             13,   8,   8,  10,  13,   0,   2,  -7,
             0,   0,   0,   0,   0,   0,   0,   0, ]
    base = 6
    for i in range(64):
        weights[base + i] = mgpwn[i]
        weights[base + i + 317] = egpwn[i]
    base += 64
    mgkn = [
        -137, -52, -66,   6,
        -45, -17,  67,  30,
        -2,  66,  83,  74,
        6,  18,  44,  45,
        -10,  12,  18,  20,
        -20,   8,  14,  14,
        -24, -34,   3,  -2,
        -64, -20, -43, -25,
    ]
    egkn = [
        -78, -50, -20, -30,
        -38, -16, -25,  -6,
        -32, -20,   0,   4,
        -18,   6,  16,  22,
        -18,  -1,  16,  20,
        -22, -12,  -2,  12,
        -43, -22, -15,  -4,
        -46, -50, -20, -18,
    ]
    for i in range(32):
        weights[base + i] = mgkn[i]
        weights[base + i + 317] = egkn[i]
    base += 32
    mgbsh = [
     -18 ,   6 , -62 , -31 ,
     -36 ,  17 ,  20 ,   8 ,
      -9 ,  37 ,  46 ,  38 ,
      -3 ,   6 ,  28 ,  44 ,
      -1 ,  12 ,  12 ,  30 ,
       5 ,  16 ,  21 ,  10 ,
       2 ,  24 ,  18 ,   4 ,
     -27 , -21 , -13 , -17 ,
    ]
    egbsh = [
        -19, -19, -10,  -8, 
        -11,  -4,  -3,  -8, 
        3,  -4,   3,  -2,  
        0,   6,  11,  12,  
        -8,   0,  12,  13, 
        -14,  -5,   6,  12,
        -20, -16,  -8,   2,
        -20,  -7, -20,  -7,
    ]
    for i in range(32):
        weights[base + i] = mgbsh[i]
        weights[base + i + 317] = egbsh[i]
    base += 32
    mgrk = [
        38,  36,  20,  57, 
        36,  29,  62,  71, 
        6,  40,  36,  26, 
        -22, -10,  21,  25,
        -30, -10, -10,   4,
        -39, -15,  -8,  -7,
        -58, -11,  -4,  -5,
        -22, -25,   4,  16,
    ]
    egrk = [
        9,   9,  15,  14,
        7,  10,   8,   4,
        2,   1,   2,   4,
        3,   1,   7,   2,
        -4,  -2,   1,   0,
        -10,  -4,  -8,  -4,
        -4,  -8,  -4,  -4,
        -14,   3,  -5,  -3,
    ]
    for i in range(32):
        weights[base + i] = mgrk[i]
        weights[base + i + 317] = egrk[i]
    base += 32
    mgqn = [
        8,  22,  36,  36, 
        15,  -6,  26,  -8, 
        22,  15,  32,  18, 
        -13, -14,   0,  -8, 
        -6, -12,  -6,  -6, 
        -4,   8,  -4,  -4,
        -17,  -6,  13,   5,
        -26, -24, -17,  -2,
    ]
    egqn = [
        6,  16,  20,  27,  
        -8,  25,  28,  50, 
        -6,  12,  22,  48, 
        20,  40,  32,  51, 
        2,  34,  26,  39,  
        -6,  -8,  16,   8,
        -27, -30, -26, -16,
        -37, -24, -27, -24,
    ]
    for i in range(32):
        weights[base + i] = mgqn[i]
        weights[base + i + 317] = egqn[i]
    base += 32
    mgkg = [
        4, 66, 33, -1,
        66, 90, 48, 24,
        92, 108, 60, 23,
        115, 134, 78, 52,
        123, 142, 103, 73, 
        146, 193, 126, 90,
        208, 227, 175, 134,
        203, 245, 203, 148,
    ]
    egkg = [
        0,    33,    63,    57, 
        39,    75,    99,   101, 
        66,    97,   126,   131,
        77,   117,   129,   129, 
        72,   124,   149,   149,
        69,   129,   138,   143,
        35,    90,    87,    98,
        8,    44,    54,    58,
    ]
    for i in range(32):
        weights[base + i] = mgkg[i]
        weights[base + i + 317] = egkg[i]
    base += 32
    mgmobkn = [-22, -19, -4, -1, 1, 5, 8, 10, 12]
    mgmobbsh = [-17, -7, 6, 9, 14, 18, 20, 23, 23, 25, 29, 29, 33, 35]
    mgmobrk = [-21, -7, 1, 1, 1, 4, 8, 11, 14, 14, 14, 16, 20, 20, 21]
    mgmobqn = [-11, -4, -3, -3, 7, 8, 8, 12, 13, 19, 22, 23, 23, 23, 24, 24, 25, 25, 27, 28, 33, 38, 38, 38, 39, 40, 40, 41]
    egmobkn = [-27, -19, -10, -5, 2, 4, 6, 7, 8]
    egmobbsh = [-20, -8, -1, 4, 8, 14, 18, 19, 22, 24, 26, 28, 29, 32]
    egmobrk = [-27, -6, 8, 13, 24, 34, 35, 41, 46, 48, 54, 56, 58, 58, 59]
    egmobqn = [-16, -10, -2, 6, 13, 19, 20, 25, 26, 32, 32, 34, 41, 43, 44, 45, 46, 48, 50, 51, 51, 57, 57, 58, 61, 61, 65, 74]
    
    for i in range(9):
        weights[base + i] = mgmobkn[i]
        weights[base + i + 317] = egmobkn[i]
    base += 9
    for i in range(14):
        weights[base + i] = mgmobbsh[i]
        weights[base + i + 317] = egmobbsh[i]
    base += 14
    for i in range(15):
        weights[base + i] = mgmobrk[i]
        weights[base + i + 317] = egmobrk[i]
    base += 15
    for i in range(28):
        weights[base + i] = mgmobqn[i]
        weights[base + i + 317] = egmobqn[i]
    base += 28
    
    return weights

def computeOptimalK(positions, targets, phases, weights):
    # This function computes the optimal K parameter for the Texel tuning algorithm
    start = 0.0
    end = 10.0
    step = 1.0
    curr = start
    best = loss(convert_scores(evaluate(positions[0], phases[0], weights), curr), targets[0])
    err = None
    for i in range(10):
        curr = start - step
        while curr < end:
            curr += step
            err = loss(convert_scores(evaluate(positions[0], phases[0], weights), curr), targets[0])
            if err <= best:
                best = err
                start = curr
        end = start + step
        start= start - step
        step = step / 10
    return start

def train():
    # Load the weights
    weights = defaultWeights()
    print(weights)
    # test the save weights function
    saveWeights(weights, "weights.txt")
    # Load the training data
    features, targets, phases = loadDataset()
    input("Press enter to continue...")

    # In order to extimate the K parameter, we will use the method described in the paper
    Kext = computeOptimalK(features, targets, phases, weights)
    print("Extimated k: ",Kext)
    input("Press enter to continue...")
    print()
    # Train the weights
    weights = texel(
        weights=weights,
        positions=features,
        targets=targets,
        phases=phases,
        K=Kext,  # for now computeOptimalK is not working
        lr = 0.25,
        EPOCHS=10000
    )

    # Save the weights
    np.save("weights.npy", weights)

if __name__ == "__main__":
    train()
