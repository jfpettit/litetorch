# litetorch

If you're like me, you might find yourself reusing lots of code between projects, and this process of reusing code might get annoying. Copying and pasting (or some other hacky way of code reuse) neural net classes between files is frustrating, can easily break, and just isn't really a very clean and modular way to reuse your code. 

To help battle this in my own work, I've written a small set of model classes and packaged them neatly into one folder. They're pretty simple and should be easy to extend if something isn't quite the right fit for you, or if you just think I could have done something better. That said, if you have any improvements on the way things are implemented here, please feel free to submit a PR.

## Installation

Clone this repository, cd into it, and pip install:

```
git clone https://github.com/jfpettit/litetorch.git
cd litetorch
pip install -e .
```

You should be able to import it into Python now!

## Usage

Import it and use it like a normal library:

```python
import litetorch as lt

mlp = lt.MLP(input_size=5, output_size=2, hidden_layer_sizes=[16])
```

And you've got an MLP! Use it as you would any other Pytorch NN.

Let's say you've got some really cool pretrained model, and you want to stack an MLP on top of it:

```python
import litetorch as lt

pretrained_cool_model = YourAwesomePretrainedModel

mlp = lt.MLP(input_size=pretrained_model_output_size, output_size=you_set_this_to_your_problem, hidden_layer_sizes=[32, 16])  # or whatever hidden layer sizes you want

model_list = [pretrained_cool_model, mlp]
chain_model = lt.ModelChain(model_list) # DISCLAIMER: I still need to write code to handle recurrent nets in the ModelChain class, so only give it feedforward nets (MLPs, CNNs)
```

Now, you can use the chained model as a regular Pytorch module.

## Contributing

If you want to add an extension to this, submit a pull request with a description of what you're adding/changing and why, as well as the file you are using to test your changes. So far, I've simply been testing by doing training runs on common datasets and making sure the model converges, so it is acceptable to submit a file that does that.