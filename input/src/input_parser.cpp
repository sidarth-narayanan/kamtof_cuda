#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <cassert>

#include "fp_data_types.h"
#include "input_parser.h"

InputParser* inputs;

// list of all valid input strings
static const std::vector<std::string> reference_input_strings = {"use_gpu_solver",
                                                                 "implicit_solver",
                                                                 "Nx",
                                                                 "Ny",
                                                                 "tol_type",
                                                                 "tol_val",
                                                                 "num_iter",
                                                                 "solver_type",
                                                                 "gpu_global_range",
                                                                 "gpu_local_range"};

void error_out(const std::string& err_msg)
{
    std::cout << " ERROR : " << err_msg << std::endl;
    std::exit(EXIT_FAILURE);
}

/**
 * @brief InputParser: Constructor for parsing input file data.
 * @param input_filename: Name of input file
*/
InputParser::InputParser(std::string input_file) : input_filename(input_file)
{
    this->read_inputs(this->input_filename);
}

/**
 * @brief InputParser: Constructor for setting default values for input options in the absence of an input file.
 * @param input_filename: Name of input file
*/
InputParser::InputParser() {}

/**
 * @brief read_inputs: Read input file and store data in an input_struct
 * @param input_filename: Name of input file
*/
void InputParser::read_inputs(const std::string& input_filename)
{
    // total number of valid inputs contained in the input data struct
    const int ninputs = reference_input_strings.size();

    // preparing to read input file line-by-line
    std::ifstream file(input_filename);

    // initializing variables
    std::string line, input_str;
    int input_count = 0;

    if (file.is_open())
    {
        while (std::getline(file,line))
        {
            input_str = find_input_option(reference_input_strings, line, input_count);
            parse_input_value(input_str, line);
        }
        file.close();
    }
    else
    {
        std::string err_msg = "Unable to open input file (" + input_filename + ").";
        error_out(err_msg);
    }

    if (input_count != ninputs)
    {
        std::string err_msg = "Number of valid inputs found (" + std::to_string(input_count) + ") in the input file (";
        err_msg = err_msg + input_filename + ") are inconsistent with the number of expected input values.";
        error_out(err_msg);
    }

    return;
}

/**
 * @brief find_input_option: For a given line in the input file, return the corresponding
 *        input option.
 * @param input_strings: Vector of strings containing all valid input options.
 * @param ninputs      : Number of valid inputs
 * @param line         : std::string containing a line from the input file.
 * @param input_count  : Running count of number of valid input options found.
*/
std::string InputParser::find_input_option(const std::vector<std::string>& input_strings,
                                           const std::string& line,
                                           int& input_count)
{
    std::string local_string = "NULL";

    // loop over all valid input options and check if an input option is contained
    // on this line
    for (int iinput = 0; iinput < input_strings.size(); iinput++)
    {
        // store input option in a local variable
        local_string = input_strings[iinput];

        // find position at which the input option occurs
        size_t pos = line.find(local_string);

        if (pos != std::string::npos)
        {
            input_count += 1;
            return local_string;
        }
    }

    size_t index = line.find(":");
    std::string option = line.substr(0,index);
    std::string err_msg = "Input option " + option +" is invalid. Please note input options are case-sensitive.";
    error_out(err_msg);

    return local_string;
}

/**
 * @brief find_input_option: For a given line in the input file, return the corresponding
 *        input option.
 * @param input_string: Vector of strings containing all valid input options.
 * @param line        : std::string containing a line from the input file.
*/
void InputParser::parse_input_value(std::string& input_string,
                                    std::string& line)
{
    // get length of input string
    int input_str_length = input_string.length();

    // get initial index of input_string in the line
    int pos = line.find(input_string);

    // initialize temporary string
    std::string temp_str = line.erase(pos, pos + input_str_length);

    // remove all spaces in the temp_str
    temp_str.erase(remove(temp_str.begin(), temp_str.end(), ' '), temp_str.end());

    // remove colon in the temp_str
    temp_str.erase(remove(temp_str.begin(), temp_str.end(), ':'), temp_str.end());

    if (input_string == "use_gpu_solver")
    {
        this->use_gpu_solver =  static_cast<bool>(std::stoi(temp_str));
    }
    else if (input_string == "implicit_solver")
    {
        this->implicit_solver =  static_cast<bool>(std::stoi(temp_str));
    }
    else if (input_string == "Nx")
    {
        this->Nx = std::stoi(temp_str);
    }
    else if (input_string == "Ny")
    {
        this->Ny = std::stoi(temp_str);
    }
    else if (input_string == "tol_type")
    {
        // convert string based input into integer value
        if (temp_str == "abs")
        {
            this->tol_type = 0;
        }
        else if (temp_str == "rel")
        {
            this->tol_type = 1;
        }
        else
        {
            std::string err_msg = "Input option not recognized for tol_type. Valid options: 0 ==> abs, 1 ==> rel";
            error_out(err_msg);
        }
    }
    else if (input_string == "tol_val")
    {
        this->tol_val = std::stod(temp_str);
    }
    else if (input_string == "num_iter")
    {
        this->num_iter = std::stoi(temp_str);
    }
    else if (input_string == "solver_type")
    {
        // convert string based input into integer value
        if (temp_str == "jacobi")
        {
            this->solver_type = 0;
        }
        else if (temp_str == "bicgstab")
        {
            this->solver_type = 1;
        }
        else
        {
            std::string err_msg = "input option not recognized for tol_val. Valid inputs: 0 ==> jacobi, 1 ==> bicgstab";
            error_out(err_msg);
        }
    }
    else if(input_string == "gpu_global_range")
    {
        this->gpu_global_range = std::stoi(temp_str);
    }
    else if(input_string == "gpu_local_range")
    {
        this->gpu_local_range = std::stoi(temp_str);
    }
    else
    {
        assert(false && "This should be checked in find_input_option function");
    }

    return;
}

/**
 * @brief print_input_struct: Output all members of the input data struct.
*/
void InputParser::print_input_struct()
{
    printf("----------------- PRINTING VALUES FROM INPUT FILES --------------\n");

    // print out all values contained in the struct
    print_variable_value(reference_input_strings[0],this->use_gpu_solver);
    print_variable_value(reference_input_strings[1],this->implicit_solver);
    print_variable_value(reference_input_strings[2],this->Nx);
    print_variable_value(reference_input_strings[3],this->Ny);
    print_variable_value(reference_input_strings[4],this->tol_type);
    print_variable_value(reference_input_strings[5],this->tol_val);
    print_variable_value(reference_input_strings[6],this->num_iter);
    print_variable_value(reference_input_strings[7],this->solver_type);
    print_variable_value(reference_input_strings[8],this->gpu_global_range);
    print_variable_value(reference_input_strings[9],this->gpu_local_range);

    printf("-----------------------------------------------------------------\n");

    return;
}

/**
 * @brief print_variable_value: print value of a variable from the input struct
*/
template <typename T>
void InputParser::print_variable_value(const std::string &varname, T& value)
{
    std::cout << varname << ": " << value << std::endl;
}
