import serial
import numpy as np

PORT = "COM12"


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
    print(buffer)


def read_output_from_STM32(serial_port):
    """
    Reads 4 bytes from the given serial port and reconstructs a float value.

    Args:
    serial_port: A serial port object.

    Returns:
    The reconstructed float value.
    """
    output = serial_port.read(1)  # Un uint8_t est sur 1 Un float IEEE 754 est codé sur 4 octets

    if len(output) != 1:
        print(len(output))
        raise ValueError("Erreur de réception : 1 octet attendus")
    
    #print("Vrai output :", output)
    #print("Octets reçus (hex) :", output.hex())  # Affiche en hexadécimal

    return output[0]



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
    for i in range(iterations):
       # print(Y_test[i])
        print(f"----- Iteration {i+1} -----")
        print(X_test[i])
        send_inputs_to_STM32(X_test[i], serial_port)
        output = read_output_from_STM32(serial_port)
        if (output == Y_test[i]):
            accuracy += 1 / iterations
            print("   Dans le mille !")
        print(f"   Expected output: {Y_test[i]}")
        print(f"   Received output: {output}")
        print(f"----------------------- Accuracy: {accuracy:.2f}\n")
        
    return accuracy


if __name__ == '__main__':
    X_test = np.load("./saved_data/X_test.npy")
    Y_test = np.load("./saved_data/Y_test.npy")
    print(X_test[7])
    print(Y_test[7])

    with serial.Serial(PORT, 115200, timeout=1) as ser:
        print("Synchronising...")
        synchronise_UART(ser)
        print("Synchronised")

        print("Evaluating model on STM32...")
        error = evaluate_model_on_STM32(100, ser)