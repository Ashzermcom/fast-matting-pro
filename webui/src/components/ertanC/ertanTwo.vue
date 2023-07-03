<template>
  <div class="bagBoX_der">
    <div class="swiper-containerTwoDer">
      <div class="swiper-wrapper">
        <div class="swiper-slide">1</div>
        <div class="swiper-slide">2</div>
        <div class="swiper-slide">3</div>
        <div class="swiper-slide">4</div>
        <div class="swiper-slide">5</div>
      </div>
    </div>
  </div>
</template>

<script>
import "swiper/css/swiper.css"; //引入swiper样式
import Swiper from "swiper"; //引入swiper

export default {
  data() {
    return {};
  },
  mounted() {
    new Swiper(".swiper-containerTwoDer", {
      speed: 5000, //匀速时间
      freeMode: true,
      loop: true,
      autoplay: {
        delay: 0,
        stopOnLastSlide: false,
        disableOnInteraction: false,
      },
      slidesPerView: 3,
      loopFillGroupWithBlank: true,
      normalizeSlideIndex: true,

      // autoplay: {
      //   disableOnInteraction: false,
      //   delay: 0,
      // },
      // speed: 5000,
      // // loop: true,
      // slidesPerView: 4,

      // spaceBetween: 30,
      // navigation: {
      //   nextEl: ".next",
      //   prevEl: ".prev",
      // },
      // autoplay: {
      //   delay: 3000, //3秒切换一次
      //   disableOnInteraction: false, },

      // scrollbar: {
      //   el: ".swiper-scrollbar",
      // },
    });
  },
};
</script>

<style lang="scss" scoped>
::v-deep .swiper-container-free-mode > .swiper-wrapper {
  -webkit-transition-timing-function: linear; /*之前是ease-out*/
  -moz-transition-timing-function: linear;
  -ms-transition-timing-function: linear;
  -o-transition-timing-function: linear;
  transition-timing-function: linear;
  margin: 0 auto;
}
.bagBoX_der {
  height: 20vh;
  height: 100%;
  border: 1px solid black;
  border-radius: 8px;
}

.swiper-containerTwoDer {
  width: 100%;
  height: 100%;
  position: relative;
  .swiper-wrapper {
    display: flex;
    justify-content: space-between;
  }

  .swiper-slide:first-child {
    margin-left: 0%;
  }
  .swiper-slide {
    position: relative;
    align-items: center;
    width: 100%;
    height: 100%;
    margin-left: 20px;
    background-color: rgb(100, 179, 147);
    img {
      box-shadow: 0 2px 3px 0 rgba(0, 0, 0, 0.2),
        0 6px 20px 0 rgba(0, 0, 0, 0.19);

      width: 100%;
      height: 100%;
    }
  }
}
</style>