import math
import re
import random
import sys



#Set of global variables from command line

activation=sys.argv[1]
training_alg=sys.argv[2]
ground_file=sys.argv[3]
distribution=sys.argv[4]
num_train=sys.argv[5]
num_test=sys.argv[6]
epsilon=sys.argv[7]
n=0
threshold=15


ground_function=""
ground_function_str=""
weights=[]


training_dataset=[]
testing_dataset=[]


#Activation function to be used by perceptron
def predict(input_vector):
    global threshold
    sum=0
    if activation=="threshold":


        # Taking dot of weights vector and input vector
        for i in range(0, len(input_vector)):
            sum = sum + weights[i] * input_vector[i]
        print(str(sum) + " : " + str(threshold))
        if sum<threshold:
            return 0
        else:
            return 1
    elif activation=="tanh":
        return (1/2)+(math.tanh((sum-threshold)/2)/2)
    elif activation=="relu":
        return max(0,sum-threshold)

#Validate NBF

def validate_NBF(line):

    if re.match("([+|-]\d+(\s+(:?AND|OR)\s+\d+)*)",line,re.X):
        return True
    else:
        return False

#Validate TF
def validate_TF(line1,line2):

    if re.match("[+|-]\d+", line1, re.X) and re.match("([+|-]\d+(\s+[+|-]\d+)*)", line2, re.X):
        return True
    else:
        return False

#Generating n bit vector
def generate_n_bit_vector(n):
    vector=[]

    for i in range(0,n):
        vector.append(random.randint(0,1))

    return vector

#Generating n bit floating vector
def generate_n_bit_fvector(n):
    vector = []
    sum=0
    for i in range(0, n):
        a=random.random()
        vector.append(a)
        sum=sum+a

    print(vector)
    print(sum)

    for i in range(0,len(vector)):
        vector[i]=vector[i]/sum

    return vector


#Determining Distribution

#Actual prediction using ground function NBF or TF

def determine_gf():

    global ground_function
    global ground_function_str

    try:
        file = open("ground_file.txt", "r")
        lines=file.readlines()

        if lines[0].replace('\n','')=="NBF":
            ground_function="NBF"
            ground_function_str=lines[1].replace('\n','').strip()
        elif lines[0].replace('\n','')=="TF":
            ground_function="TF"
            ground_function_str=lines[2]

    except IOError:
        print("ground_file.txt cannot be opened")


def evaluate(str, x):

    if int(str) > 0:
        return x[int(str)-1]
    else:
        return abs(x[abs(int(str))]-1)

def actual_pred_NBF(input_vector):

    global ground_function_str

    li = ground_function_str.split()

    value = evaluate(li[0],input_vector)
    for i in range(1, len(li), 2):

        if li[i] == 'OR':

            value = value or evaluate(li[i + 1], input_vector)
        elif li[i] == 'AND':

            value = value and evaluate(li[i + 1], input_vector)

    return value

def actual_pred_TF(input_vector):

    global ground_function_str

    li = ground_function_str.split()

    file_TF = open("TF.txt", "r")
    d = file_TF.readlines()[1]

    value = 0

    for i in range(0, len(li)):
        value = value + int(li[i]) * input_vector[i]


    if value > int(d):
        return 1
    else:
        return 0



#def determine_distribution():

#Perceptron Update rule for NBF
def update_perceptron_NBF(input_vector):
    global threshold
    global weights

    prediction=predict(input_vector)

    actual_prediction=actual_pred_NBF(input_vector)

    #Check for prediction

    #False Positive
    if prediction>actual_prediction:

        for i in range(0, len(input_vector)):
            weights[i]=weights[i]-input_vector[i]
        threshold=threshold+1

        return "update",prediction
    #False Negative
    elif prediction<actual_prediction:
        for i in range(0, len(input_vector)):
            weights[i]=weights[i]+input_vector[i]
        threshold=threshold-1

        return "update",prediction
    else:
        return "no update",prediction

def update_perceptron_TF(input_vector):
    global threshold
    global weights

    prediction = predict(input_vector)

    actual_prediction = actual_pred_TF(input_vector)

    # Check for prediction

    # False Positive
    if prediction > actual_prediction:

        for i in range(0, len(input_vector)):
            weights[i] = weights[i] - input_vector[i]
        threshold = threshold + 1

        return "update", prediction
    # False Negative
    elif prediction < actual_prediction:
        for i in range(0, len(input_vector)):
            weights[i] = weights[i] + input_vector[i]
        threshold = threshold - 1

        return "update", prediction
    else:
        return "no update", prediction


#Winnow Update Rule for NBF
def update_winnow_NBF(input_vector):
    global weights
    global threshold

    prediction=predict(input_vector)
    actual_prediction = actual_pred_NBF(input_vector)

    alpha=random.randint(2,10)

    # False Positive
    if prediction > actual_prediction:

        for i in range(0, len(input_vector)):
            weights[i] = weights[i]*pow(alpha,-1*input_vector[i])

        return "update", prediction

    # False Negative
    elif prediction < actual_prediction:
        for i in range(0, len(input_vector)):
            weights[i] = weights[i] * pow(alpha, 1 * input_vector[i])

        return "update", prediction
    else:
        return "no update", prediction


def update_winnow_TF(input_vector):
    global weights
    global threshold

    prediction=predict(input_vector)
    actual_prediction = actual_pred_TF(input_vector)

    alpha = random.randint(2, 10)

    # False Positive
    if prediction > actual_prediction:

        for i in range(0, len(input_vector)):
            weights[i] = weights[i] * pow(alpha, -1 * input_vector[i])

        return "update", prediction

    # False Negative
    elif prediction < actual_prediction:
        for i in range(0, len(input_vector)):
            weights[i] = weights[i] * pow(alpha, 1 * input_vector[i])

        return "update", prediction
    else:
        return "no update", prediction

#Calculate n

def calculate_n():
    global ground_function
    global n

    print("Inside calculate_n()")
    try:
        file = open("ground_file.txt", "r")


        if ground_function=="NBF":
            str = file.readlines()[1]
            li = [int(s) for s in str.replace('+', ' ').replace('-', '').split() if s.isdigit()]
            n=max(li)
        elif ground_function=="TF":
            str = file.readlines()[2]
            li = [int(s) for s in str.replace('+', ' ').replace('-', '').split() if s.isdigit()]
            n = len(li)

    except IOError:
        print("File ground_file.txt cannot be opened")

#Train perceptron
def train_Perceptron():

    global ground_function

    print("Inside Train_Perceptron()")


    print("Ground function: "+ground_function_str)


    if ground_function=="NBF":

        if training_alg=="perceptron":
            print("Training Algorithm: "+training_alg)

            for i in range(0,int(num_train)):
                status,prediction=update_perceptron_NBF(training_dataset[i])

                for j in range(0,len(training_dataset[i])):
                    print(" "+str(training_dataset[i][j]),end='')

                print(" : "+str(prediction)+" "+status+"\n")

        elif training_alg == "winnow":
            for i in range(0, int(num_train)):
                status, prediction= update_winnow_NBF(training_dataset[i])

                for j in range(0, len(training_dataset[i])):
                    print(" " + str(training_dataset[i][j]), end='')

                print(" : " + str(prediction) + " " + status + "\n")

    elif ground_function=="TF":

        if training_alg=="perceptron":
            print("Training Algorithm: "+training_alg)

            for i in range(0,int(num_train)):
                status,prediction=update_perceptron_TF(training_dataset[i])

                for j in range(0,len(training_dataset[i])):
                    print(" "+str(training_dataset[i][j]),end='')

                print(" : "+str(prediction)+" "+status+"\n")

        elif training_alg == "winnow":
            for i in range(0, int(num_train)):
                status, prediction = update_winnow_TF(training_dataset[i])

                for j in range(0, len(training_dataset[i])):
                    print(" " + str(training_dataset[i][j]), end='')

                print(" : " + str(prediction) + " " + status + "\n")

#Test perceptron

def test_perceptron():
    print("Inside test_perceptron()")
    no_of_errors=0

    global ground_function

    if ground_function == "NBF":
        for i in range(0,int(num_test)):
            vector=testing_dataset[i]
            prediction=predict(vector)
            actual_prediction=actual_pred_NBF(vector)

            for j in range(0,len(vector)):
                print(" "+str(vector[j]),end='')

            print(" : "+str(prediction)+" : "+str(actual_prediction)+" : "+str(abs(prediction-actual_prediction)))

            if prediction!=actual_prediction:
                no_of_errors+=1

    elif ground_function == "TF":
        for i in range(0,int(num_test)):
            vector=testing_dataset[i]
            prediction=predict(vector)
            actual_prediction=actual_pred_TF(vector)

            for j in range(0,len(vector)):
                print(" "+str(vector[j]),end='')

            print(" : "+str(prediction)+" : "+str(actual_prediction)+" : "+str(abs(prediction-actual_prediction)))

            if prediction!=actual_prediction:
                no_of_errors+=1

    avg_error=no_of_errors/int(num_test)

    print("\n\nAverage Error: "+str(avg_error))
    print("Epsilon: "+str(epsilon))

    if avg_error< float(epsilon):
        print("TRAINING SUCCEEDED")
    else:
        print("TRAINING FAILED")

#Validate ground_file

def validate_ground_file():
    print("Inside validate_ground_file()")

    try:
        file=open("ground_file.txt")

        lines=file.readlines()

        if lines[0].replace('\n','')=="NBF":
            print(lines[1])
            return validate_NBF(lines[1])
        elif lines[0].replace('\n','')=="TF":
            return validate_TF(lines[1],lines[2])



    except IOError :
        print("Ground file cannot be opened")

#Driver of the program


if not validate_ground_file():
    print("NOT PARSEABLE")
    exit()

determine_gf()

print("Ground function :"+ground_function)
print("Ground function str: "+ground_function_str)

calculate_n()

print("Value of n:"+str(n))

c = 4
k = float(1) / float(2 * c)
delta = float(1) / float(1000)

m1 = (144 * (n + 1) ** 2) / (25 * (k * float(epsilon)) ** 2)

m = max((-144 * math.log(delta / float(2))), m1)
print("Value of m: "+str(int(m)))

#Populating training and testing dataset

#If ground function is NBF, ignore the value of the variable distribution
if ground_function=="NBF":
    for i in range(0,int(num_train)):
        training_dataset.append(generate_n_bit_vector(n))
    for i in range(0,int(num_test)):
        testing_dataset.append(generate_n_bit_vector(n))

#If ground file is TF, check the value of variable distribution
elif ground_function=="TF":
    if distribution=="bool":
        for i in range(0, int(num_train)):
            training_dataset.append(generate_n_bit_vector(n))
        for i in range(0, int(num_test)):
            testing_dataset.append(generate_n_bit_vector(n))
    elif distribution=="sphere":
        for i in range(0, int(num_train)):
            training_dataset.append(generate_n_bit_fvector(n))
        for i in range(0, int(num_test)):
            testing_dataset.append(generate_n_bit_fvector(n))

print("Length of training dataset:"+str(len(training_dataset)))

print("Length of testing dataset:" + str(len(testing_dataset)))


for i in range(0, n):
    weights.append(0)

print("Weights before training: ")
print(weights)

train_Perceptron()

print("Weights after training: ")
print(weights)

test_perceptron()

print("Value of m:"+str(int(m)))
print("Threshold: "+str(threshold))




