const images = [
    "/static/assets/americanFlag.jpg",
    "/static/assets/breakingNews.jpg",
    "/static/assets/newspaper.jpg"
];

let current = 0;
const header = document.getElementById("header");
function changeBackground() {
    header.style.backgroundImage =
        `url('${images[current]}')`;
    current = (current + 1) % images.length;
}
// Set initial image
changeBackground();
// Change every five seconds
setInterval(changeBackground, 5000);