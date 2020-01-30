# Maps Symbols
To create a custom map for either the single agent and the multi-agent environment, you only have to create a txt file with the grid structure and then pass the file name as an argument to the class constructor.

For the single agent environment, allowed symbols are:
* 0 -> empty cell
* x -> agent starting position
* G -> goal position
* 1 -> obstacle
* B -> bomb (gives a penalty)

For the multi-agent one:
* 0 -> empty cell
* P -> pursuer starting position (there have to be multiple)
* E -> escaper initial position
* 1 -> obstacle

### Author
*Castellini Jacopo*
