// ============================================================
// Academic Project Page - Custom Scripts
// ============================================================

$(document).ready(function () {
  "use strict";

  // --- Navbar burger toggle for mobile ---
  $(".navbar-burger").click(function () {
    $(".navbar-burger").toggleClass("is-active");
    $(".navbar-menu").toggleClass("is-active");
  });

  // --- Initialize carousel(s) ---
  // Uncomment and configure if you use a carousel:
  //
  // var options = {
  //   slidesToScroll: 1,
  //   slidesToShow: 3,
  //   loop: true,
  //   infinite: true,
  //   autoplay: false,
  //   autoplaySpeed: 3000,
  // };
  //
  // if (document.getElementById("results-carousel")) {
  //   bulmaCarousel.attach("#results-carousel", options);
  // }

  // --- Lazy-load videos (optional) ---
  // You can add intersection observer logic here to only
  // load videos when they come into view.
});
