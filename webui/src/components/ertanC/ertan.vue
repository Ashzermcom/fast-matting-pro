<template>
  <div class="bagBoX_der">
    <div class="swiper-containerDER">
      <div class="swiper-wrapper">
        <div class="swiper-slide">
          <div class="mainLIst_Boxder">
            <!-- list -->
            <div class="list_sertrwer">
              <div class="list_serS_er">
                <!-- line -->
                <div></div>
                <div></div>
                <div></div>
              </div>
              <div class="right_serS_er"></div>
            </div>
            <div class="list_sertrwer">
              <div class="list_serS_er">
                <!-- line -->
                <div></div>
                <div></div>
                <div></div>
              </div>
              <div class="right_serS_er"></div>
            </div>
            <div class="list_sertrwer">
              <div class="list_serS_er">
                <!-- line -->
                <div></div>
                <div></div>
                <div></div>
              </div>
              <div class="right_serS_er"></div>
            </div>
          </div>
        </div>
        <div class="swiper-slide">
          <div class="mainLIst_Boxder">
            <!-- list -->
            <div class="list_sertrwer">
              <div class="list_serS_er">
                <!-- line -->
                <div></div>
                <div></div>
                <div></div>
              </div>
              <div class="right_serS_er"></div>
            </div>
            <div class="list_sertrwer">
              <div class="list_serS_er">
                <!-- line -->
                <div></div>
                <div></div>
                <div></div>
              </div>
              <div class="right_serS_er"></div>
            </div>
            <div class="list_sertrwer">
              <div class="list_serS_er">
                <!-- line -->
                <div></div>
                <div></div>
                <div></div>
              </div>
              <div class="right_serS_er"></div>
            </div>
          </div>
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
    new Swiper(".swiper-containerDER", {
      // direction: "vertical", // 垂直切换选项
      // loop: true, // 循环模式选项
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
  border: 1px solid black;
  border-radius: 8px;
}

.swiper-containerDER {
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
    background-color: rgb(215, 99, 32);
    img {
      box-shadow: 0 2px 3px 0 rgba(0, 0, 0, 0.2),
        0 6px 20px 0 rgba(0, 0, 0, 0.19);

      width: 100%;
      height: 100%;
    }
  }
}
.mainLIst_Boxder {
  width: 92%;
  margin: 0 auto;
  .list_sertrwer {
    display: flex;
    height: 80px;
    justify-content: space-between;
    .list_serS_er {
      width: 60%;
      height: 100%;
      border: 1px solid black;
      div:nth-of-type(1) {
        width: 90%;
        height: 15px;
        margin: 8px 0 0 10px;
        background-color: rgb(96, 77, 77);
      }
      div:nth-of-type(2) {
        width: 40%;
        height: 15px;
        margin: 8px 0 0 10px;
        background-color: rgb(96, 77, 77);
      }
      div:nth-of-type(3) {
        width: 20%;
        height: 15px;
        margin: 8px 0 0 10px;
        background-color: rgb(96, 77, 77);
      }
    }
    .right_serS_er {
      width: 35%;
      height: 100%;
      border: 1px solid black;
    }
  }
}
</style>