import serial
import numpy as np

PORT = "COM12" # Change this to the port where the STM32 is connected

def synchronise_UART(serial_port):
    """
    Synchronizes the UART communication by sending a byte and waiting for a response.
    Args:
        serial_port (serial.Serial): The serial port object to use for communication.
    Returns:
        None
    """
    while (1):
        serial_port.write(b"\xAB")
        ret = serial_port.read(1)
        if (ret == b"\xCD"):
            serial_port.read(1)
            while serial_port.in_waiting > 0:
                serial_port.read(1)  # Lire un octet et l'ignorer
            break


def send_inputs_to_STM32(inputs, serial_port):
    """
    Sends a numpy array of inputs to the STM32 microcontroller via a serial port.

    Args:
        inputs (numpy.ndarray): The inputs to send to the STM32.
        serial_port (serial.Serial): The serial port to use for communication.

    Returns:
        None
    """
    inputs = inputs.astype(np.float32)
    buffer = b""
    for x in inputs:
        buffer += x.tobytes()
    serial_port.write(buffer)


def read_output_from_STM32(serial_port):
    """
    Reads only 1 bytes from the given serial port and recuperate directly a float value. The interest to use a int instead
    of a float is to save transmission time and to be able to use the int value directly from the STM32 without any reconstruction.

    Args:
    serial_port: A serial port object.

    Returns:
    The value.
    """
    output = serial_port.read(1)  # Un uint8_t est sur 1

    if len(output) != 1:
        raise ValueError("Erreur de réception : 1 octet attendu mais {} reçu(s).".format(len(output)))

    return output[0]

#The output of the card is analyzed to show the user the type of failure predicted (or to say that there is no problem).
def analyse_output(output): 
    if output == 0:
        return "No failure"
    elif output == 1:
        return "Failure TWF"
    elif output == 2:
        return "Failure HDF"
    elif output == 3:
        return "Failure PWF"
    elif output == 4:
        return "Failure OSF"



def evaluate_model_on_STM32(iterations, serial_port):
    """
    Evaluates the accuracy of a machine learning model on an STM32 device.

    Args:
        iterations (int): The number of iterations to run the evaluation for.
        serial_port (Serial): The serial port object used to communicate with the STM32 device.

    Returns:
        float: The accuracy of the model, as a percentage.
    """
    accuracy = 0
    for i in range(iterations):#The number of iterations is defined by the user in the main function
        print(f"----- Iteration {i+1} -----")
        send_inputs_to_STM32(X_test[i], serial_port)
        output = read_output_from_STM32(serial_port)
        if (output == Y_test[i]):
            accuracy += 1 / iterations
            print(f"   Output: {analyse_output(output)}")
        else:
            print("_______________ERROR_______________")
            print(f"   Expected output: {Y_test[i]}")
            print(f"   Received output: {output}")
        print(f"----------------------- Accuracy: {accuracy:.2f}\n")

    return accuracy

# Main function
if __name__ == '__main__':

     #The data is loaded from the saved_data folder
    X_test = np.load("./saved_data/X_test.npy") # Change to X.npy if you want to test the raw (balanced) data
    Y_test = np.load("./saved_data/Y_test.npy") # Change to Y.npy if you want to test the raw (balanced) data

    with serial.Serial(PORT, 115200, timeout=1) as ser:
        print("Synchronising...")
        synchronise_UART(ser)
        print("Synchronised")

        print("Evaluating model on STM32...")
        error = evaluate_model_on_STM32(500, ser)#The number of iterations is defined by the user here