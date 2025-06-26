    .::::     .:::::::  .::: .::::::.::      .::    .::   
  .::    .::  .::    .::     .::     .::   .::   .::   .::
.::        .::.::    .::     .::      .:: .::   .::       
.::        .::.:::::::       .::        .::     .::       
.::        .::.::            .::      .:: .::   .::       
  .::     .:: .::            .::     .::   .::   .::   .::
    .::::     .::            .::    .::      .::   .::::    .::

## Key Functions

* Transforming SMILES into .xyz and calculating descriptors for XC functionals optimization
* Machine Learning Model for XC functionals optimization when calculating the charge-transfer excited or local excited state of singe molecular systems in TDDFT
* Scripts for batch generation of Gaussian tasks before and after optimization of XC functionals

## How To Use

To clone and run this application, you'll need some python packages installed on your computer. From your command line:

```bash
# Clone this repository
$ git clone https://github.com/STOKES-DOT/code_ml_dft

# Go into the repository
$ cd code_ml_dft

```
## XC functionals optimization

To optimize the XC functionals of a given molecule, you can use the following code in example folder:

```bash
# Go into the example folder
$ cd example

# You can use the following code to optimize the XC functionals of a given molecule with a given initial guess
$ python optimize_xc.py --smiles "C" --xc "LC-wPBE" --basis "6-31G" --initial_guess "LC-wPBE"

```
## Transforming SMILES into .xyz and calculating descriptors for XC functionals optimization
This code will generate a .xyz file for the given molecule, and then optimize the XC functionals using the given initial guess. The optimized XC functionals will be saved in the same folder with the initial guess.
> **Note**
> Here, We give two XC funationals optimization code. If you want to create a new model for optimizing other XC functional, Please refer to our article in XXX (Will be pulished in a few time). 
> The final stacking model is not training because of the limit in Github's storage space. You need to finish training it by youself. Besides, each base-learners could also be used for optimization with a lower performance.
> The packge needed for training stacking model is listed in packge.txt.

## Email

If you liked using this model or it has any questions about these codes or our theory details, I'd like you send me an email at <jiaoyuan24@mails.ucas.ac.cn> about anything you'd want to say about these codes. I'd really appreciate it!

## Credits

This code uses the following open source packages (We have listed in the project. You can install them accroding to your need.):


## Support

If you like this project and think it has helped in any way, Sent me 2.5$ for a cup of coffee! Just a joke, but it's always nice to give back to the community. Have a great day!


## DFT computational program for strongly correlated system based on neural network

This project is just an idea. We will keep updating it in the future. If you have any questions or suggestions, please feel free to contact me.

## License

UCAS,SAIS

---


