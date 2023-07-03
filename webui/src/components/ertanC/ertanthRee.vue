<template>
  <div class="bagBoX_der">
    <div class="swiper-containTDer">
      <div class="swiper-wrapper">
        <div class="swiper-slide">
          <!-- <img
            @click="care(3)"
            src="../../assets/falseImg/textDer_ser.jpg"
            alt=""
          /> -->
          <!-- <img @click="care(1)" src="../../assets/falseImg/fefqw.png" alt="" /> -->
        </div>
        <div class="swiper-slide">
          <!-- <img
            @click="care(3)"
            src="../../assets/falseImg/textDer_ser.jpg"
            alt=""
          /> -->
          <!-- <img @click="care(2)" src="../../assets/falseImg/Derine.png" alt="" /> -->
        </div>
        <div class="swiper-slide">
          <!-- <img
            @click="care(3)"
            src="../../assets/falseImg/textDer_ser.jpg"
            alt=""
          /> -->
        </div>
        <div class="swiper-slide">
          <!-- <img
            @click="care(3)"
            src="../../assets/falseImg/textDer_ser.jpg"
            alt=""
          /> -->
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
    this.textDer();
  },
  methods: {
    textDer() {
      new Swiper(".swiper-containTDer", {
        //   speed: 8000, //匀速时间
        freeMode: true,
        loop: true,
        spaceBetween: 20,
        slidesPerView: 1,
        autoplay: {
          delay: 5000,
          stopOnLastSlide: false,
          disableOnInteraction: false,
        },
        coverflowEffect: {
          rotate: 30, //slide做3d旋转时Y轴的旋转角度。默认50。
          stretch: -4, //每个slide之间的拉伸值，越大slide靠得越紧。 默认0。
          depth: 60, //slide的位置深度。值越大z轴距离越远，看起来越小。 默认100。
          modifier: 2, //depth和rotate和stretch的倍率，相当于depth*modifier、rotate*modifier、stretch*modifier，值越大这三个参数的效果越明显。默认1。
          slideShadows: true, //开启slide阴影。默认 true。
        },
        // loopFillGroupWithBlank: true,
        // normalizeSlideIndex: true,

        // autoplay: {
        //   disableOnInteraction: false,
        //   delay: 0,
        // },
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

    care(item) {
      //  通过 $emit 来向父组件发射事件（信息）,自定义事件
      this.$emit("itemclick", item);
    },
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
  height: 100%;
  width: 100%;
  position: relative;
  border-radius: 8px;
}

.swiper-containTDer {
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
    position: relative;
    img {
      // box-shadow: 0 2px 3px 0 rgba(0, 0, 0, 0.2),
      //   0 6px 20px 0 rgba(0, 0, 0, 0.19);
      // border: 1px solid black;
      width: 100%;
      height: 100%;
      border-radius: 10px;
    }
  }
}
</style>