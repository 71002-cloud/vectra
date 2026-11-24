height = 10
length = 7

board = []

for (let i = 0; i < height; i++) {
    for (let j = 0; j < length; j++) {
        let square = document.createElement("div");
        square.classList.add("square");
        (i + j) % 2 === 0 ? square.classList.add("black") : square.classList.add("white");
        board.push(square);
    }
}

for (let i = 0; i < board.length; i++) {
    document.getElementsByClassName("board")[0].appendChild(board[i]);
}