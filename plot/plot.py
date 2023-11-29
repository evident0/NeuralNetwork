
import matplotlib.pyplot as plt
import matplotlib

def read_error_list(filename):
    with open(filename, 'r') as f:
        return f.readlines()


def read_data(filename):
    list = []
    with open(filename, 'r') as f:
        for line in f.readlines():
            tpl = line.strip('\n').split(',')
            new_tpl = (float(tpl[0]), float(tpl[1]), tpl[2], True if tpl[3] == 'true' else False)
            list.append(new_tpl)
    return list

def plot_results(testDataTupleList):
    for x,y,category,is_correct in testDataTupleList:
        if is_correct:
            if category == 'C1':
                plt.plot(x, y, c='y', marker='x')
            elif category == 'C2':
                plt.plot(x, y, c='b', marker='x')
            elif category == 'C3':
                plt.plot(x, y, c='g', marker='x')
        else:
            plt.plot(x, y, c='r', marker='x')

    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Test Data')
    plt.show()


matplotlib.use('TkAgg')

#plot the test data
testDataTupleList = read_data('testDatasetResults.txt')
plot_results(testDataTupleList)

#plot the error list
desired_array = [float(numeric_string) for numeric_string in read_error_list('errorList.txt')]

plt.plot(desired_array)
plt.xlabel('Epoch')
plt.ylabel('Total Error')

plt.show()
