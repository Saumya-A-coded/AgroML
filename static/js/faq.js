document.addEventListener("DOMContentLoaded", function() {
    const questions = document.querySelectorAll(".faq-question");

    questions.forEach(q => {
        q.addEventListener("click", () => {
            q.classList.toggle("active");
            let answer = q.nextElementSibling;
            let symbol = q.querySelector("span");

            if (answer.style.display === "block") {
                answer.style.display = "none";
                symbol.textContent = "+";
            } else {
                answer.style.display = "block";
                symbol.textContent = "-";
            }
        });
    });
});

// document.addEventListener("DOMContentLoaded", function() {
//     let questions = document.querySelectorAll(".faq-question");
//     questions.forEach(q => {
//         q.addEventListener("click", () => {
//             let answer = q.nextElementSibling;
//             answer.style.display = (answer.style.display === "block") ? "none" : "block";
//         });
//     });
// });
