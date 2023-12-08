import random
import numpy as np


class Support:
    """
    Support class with utility methods for Hamming Code operations.
    """

    @staticmethod
    def create_random_vector():
        """
        Generate a random 4-bit vector.

        Returns:
            np.ndarray: Random 4-bit vector.
        """
        random_vector = np.random.randint(0, 2, size=4)
        return random_vector

    @staticmethod
    def check_input(x, input_length):
        """
        Check if the input satisfies the requirements for Hamming Code operations.

        Args:
            x (int, list, str, np.ndarray): Input to be validated.
            input_length (int): Expected length of the input array.

        Returns:
            np.ndarray: Validated and converted input array.

        Raises:
            TypeError: If the input type is not int, list, str, or np.ndarray.
            ValueError: If input requirements are not met.
        """
        # Check if x is an integer, a list, or a NumPy array
        if not isinstance(x, (int, list, np.ndarray, str)):
            raise TypeError("The input must be an integer, a list, a string, or a NumPy array.")

        # If x is an integer, convert it to a list of its digits
        if isinstance(x, int):
            x = [int(digit) for digit in str(x)]

        # Remove white spaces if x is a string
        if isinstance(x, str):
            x = x.replace(" ", "")
            if not x.isdigit():
                raise ValueError("String input must contain only digits.")
            # Convert string of digits to list of integers
            x = [int(digit) for digit in x]

        # Convert x to a NumPy array if it's not already
        if not isinstance(x, np.ndarray):
            try:
                x = np.array(x)
            except Exception as e:
                raise ValueError(f"Failed to convert input to a NumPy array: {e}")

        # Check if the array has the specified input length
        if x.size != input_length:
            raise ValueError(f"The input array must have a size of {input_length}.")

        # Check if the elements are integers
        if not np.issubdtype(x.dtype, np.integer):
            raise ValueError("The input array must contain only integers.")

        # Check if the elements are either 0 or 1
        if not np.all(np.isin(x, [0, 1])):
            raise ValueError("The input array must contain only 0s and 1s.")

        return x

    @staticmethod
    def bitflip_rand(x, number_of_bits_to_flip):
        """
        Flip a specified number of random bits in the input array.

        Args:
            x (np.ndarray): Input array.
            number_of_bits_to_flip (int): Number of bits to flip.

        Returns:
            np.ndarray: Input array with flipped bits.

        Raises:
            ValueError: If the number_of_bits_to_flip is greater than 2.
        """
        # Check if the input has valid length
        x = Support.check_input(x, 7)
        # Flip the number of bits specified in number_of_bits_to_flip
        if number_of_bits_to_flip > 2:
            raise ValueError("The Hamming Code only allows detecting at most two bitflips.")
        else:
            flipped_bits = set()

            while True:
                i = random.randint(0, 6)
                if i not in flipped_bits:
                    flipped_bits.add(i)
                    x[i] = 1 - x[i]
                if len(flipped_bits) == number_of_bits_to_flip:
                    break
        return x

    @staticmethod
    def bitflip_specific(x, bit_to_flip):
        """
        Flip a specific bit in the input array.

        Args:
            x (np.ndarray): Input array.
            bit_to_flip (int): Index of the bit to flip.

        Returns:
            np.ndarray: Input array with the specified bit flipped.

        Raises:
            ValueError: If bit_to_flip is greater than 6 (out of bounds).
        """
        # User can choose which bit should be flipped by giving the index of the bit
        x = Support.check_input(x, 7)
        if bit_to_flip > 6:
            raise ValueError("The codeword has only 7 bits.")
        else:
            x[bit_to_flip] = 1 - x[bit_to_flip]
        return x


class Hamming:
    """
    Class for encoding, decoding, and checking Hamming codes.
    """
    # GeneratorMatrix (G)
    G = np.array([[1, 1, 0, 1],
                  [1, 0, 1, 1],
                  [1, 0, 0, 0],
                  [0, 1, 1, 1],
                  [0, 1, 0, 0],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1]])
    
    # ParityCheckMatrix (H)
    H = np.array([[1, 0, 1, 0, 1, 0, 1],
                  [0, 1, 1, 0, 0, 1, 1],
                  [0, 0, 0, 1, 1, 1, 1]])
    
    # DecoderMatrix (R)
    R = np.array([[0, 0, 1, 0, 0, 0, 0],
                  [0, 0, 0, 0, 1, 0, 0],
                  [0, 0, 0, 0, 0, 1, 0],
                  [0, 0, 0, 0, 0, 0, 1]])

    @staticmethod
    def encoder(input):
        """
        Encode a 4-bit vector into a 7-bit codeword using the Generator Matrix (G).

        Args:
            input (int, list, str, np.ndarray): 4-bit vector to be encoded.

        Returns:
            np.ndarray: 7-bit codeword.
        """
        # Check if the input has valid length
        input = Support().check_input(input, 4)
    
        codeword = np.matmul(Hamming.G, input) % 2
        print(f'The 7-bit codeword is {codeword}.')
        return codeword
    
    @staticmethod
    def parity_check(codeword):
        """
        Perform a parity check on a 7-bit codeword using the Parity Check Matrix (H).

        Args:
            codeword (int, list, str, np.ndarray): 7-bit codeword to be checked.

        Returns:
            None
        """
        # Check if the input has valid length
        codeword = Support().check_input(codeword, 7)
        
        error_syndrome = np.matmul(Hamming.H, codeword) % 2
        print(f'The error vector is {error_syndrome}.')

        sum_error_syndrome = np.sum(error_syndrome)
        if sum_error_syndrome == 0:
            print('There was no error occurring upon code transmission.')
        elif sum_error_syndrome == 1:
            print('The sum of the error syndrome elements is 1. There was an error occurring upon code transmission in either one of the parity bits or there were two errors occurring in the data bits. Either way, the error cannot be corrected.')
        else: # sum_error_syndrome == 2 or 3
            # transpose the matrix to be able to compare our error syndrome with the rows of the transposed matrix
            transposed_parity = Hamming.H.transpose()
            # convert the matrix to a list to be able to use the index function
            list_parity = transposed_parity.tolist()
            list_parity_index = list_parity.index(error_syndrome.tolist())
            # add 1 to the index to get the correct row number
            bitflip_pos = list_parity_index + 1
            print(f'There was an error occurring upon code transmission and the error can be corrected.\nThe Bitflip occurred at the {bitflip_pos}th position.')
        return

    @staticmethod
    def decoder(codeword):
        """
        Decode a 7-bit codeword into the original 4-bit vector, correcting errors if possible.

        Args:
            codeword (int, list, str, np.ndarray): 7-bit codeword to be decoded.

        Returns:
            None
        """
        # Check if the input has valid formal requirements
        codeword = Support().check_input(codeword, 7)

        error_syndrome = np.matmul(Hamming.H, codeword) % 2
        sum_error_syndrome = np.sum(error_syndrome)

        if sum_error_syndrome == 0:
            print('There is no bitflip error in the codeword to be corrected.')
            # decoder function returns 'original' 4-bit vector when given a 7-bit codeword
            binary_decoded = np.matmul(Hamming.R, codeword) % 2
            print(f'The original 4-bit vector is {binary_decoded}.')
        elif sum_error_syndrome == 1:
            print('The sum of the error syndrome elements is 1. There is an error in either one of the parity bits or there are two errors in the data bits. In these cases, the error cannot be decoded.')
        else: # sum_error_syndrome == 2 or 3
             # transpose the matrix to be able to compare our error syndrome with the rows of the transposed matrix
            transposed_parity = Hamming.H.transpose()
            # convert the matrix to a list to be able to use the index function
            list_parity = transposed_parity.tolist()
            list_parity_index = list_parity.index(error_syndrome.tolist())
            # add 1 to the index to get the correct row number
            bitflip_pos = list_parity_index + 1
            print(f'The bit at position {bitflip_pos} was flipped.')

            # flip the bit at the position of the bitflip
            if codeword[bitflip_pos - 1] == 0:
                codeword[bitflip_pos - 1] = 1
            else:
                codeword[bitflip_pos - 1] = 0
            print(f'The corrected codeword is {codeword}.')
            binary_decoded = np.matmul(Hamming.R, codeword) % 2
            print(f'The original 4-bit vector is {binary_decoded}.')
        return