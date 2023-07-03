<template>
  <div class="bagBoX_der">
    <div class="swiper-conSerDer_E">
      <div class="swiper-wrapper">
        <div class="swiper-slide">
          <img src="../../assets/falseImg/leftIcb.png" alt="" />
        </div>
        <div class="swiper-slide">
          <img src="../../assets/falseImg/leftIcb.png" alt="" />
        </div>
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
    new Swiper(".swiper-conSerDer_E", {
      // direction: "vertical", // 垂直切换选项
      loop: true, // 循环模式选项
      slidesPerView: "auto",
      loopedSlides: 1,
      slidesPerView: 1,
      spaceBetween: 30,
      // freeMode: true,
      navigation: {
        nextEl: ".next",
        prevEl: ".prev",
      },
      autoplay: {
        delay: 3000, //3秒切换一次
        disableOnInteraction: false,

        //   pauseOnMouseEnter: true, //悬停停止切换
      },
      // 如果需要分页器
      // pagination: {
      //   el: ".swiper-pagination",
      // },

      // 如果需要前进后退按钮
      // navigation: {
      //   nextEl: ".swiper-button-next",
      //   prevEl: ".swiper-button-prev",
      // },

      // 如果需要滚动条
      scrollbar: {
        el: ".swiper-scrollbar",
      },
    });
  },
};
</script>
 
 <style lang="scss" scoped>
.bagBoX_der {
  width: 100%;
  //   height: 20vh;
  border-radius: 8px;
}

.swiper-conSerDer_E {
  width: 100%;
  height: 100%;
  position: relative;
  .swiper-wrapper {
    display: flex;
    justify-content: space-between;
  }
  .swiper-slide {
    position: relative;
    align-items: center;
    width: 100%;
    height: 100%;
    background-color: #ffffff;
    img {
      box-shadow: 0 2px 3px 0 rgba(0, 0, 0, 0.2),
        0 6px 20px 0 rgba(0, 0, 0, 0.19);

      width: 100%;
      height: 100%;
    }
  }
}
</style>