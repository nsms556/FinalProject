import React, {useState} from "react";

import axios from "axios";

import SelectGenre from "./SelectGenre";
import SelectSongs from "./SelectSongs";


function MakePlaylist() {

    const [UserData, setUserData] = useState({like: {}, dislike: {}});

    const [ViewGenre, setViewGenre] = useState(true);
    const [ViewSongs, setViewSongs] = useState(false);

    const [Tags, setTags] = useState(null);

    const onChangeState = (LikeGenre, DislikeGenre) => {

        UserData["like"]["tags"] = LikeGenre;
        UserData["dislike"]["tags"] = DislikeGenre;

        setUserData(UserData);

        setTags(LikeGenre);

        setViewGenre(!ViewGenre);
        setViewSongs(!ViewSongs);
    };

    const onFinished = (LikeSong, DislikeSong) => {

        UserData["like"]["songs"] = LikeSong;
        UserData["dislike"]["songs"] = DislikeSong;

        setUserData(UserData);

        axios.post('http://127.0.0.1:8000/playlist/recommend', UserData)
            .then(response => {
                if (response.data && response.data.success) {
                    window.location.replace("/admin/playlist");
                } else {
                    alert('데이터 전송에 실패했습니다.');
                }
            })
            .catch(error => {
                console.log(error);
            })
    };

    return (
        <>
            {
                ViewGenre &&
                <SelectGenre onChangePage={onChangeState}/>
            }
            {
                ViewSongs && Tags &&
                <SelectSongs tags={Tags} onSubmit={onFinished}/>
            }
        </>
    );
}

export default MakePlaylist;
