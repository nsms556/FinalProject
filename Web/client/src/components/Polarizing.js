import React, {useEffect, useState} from "react";

import axios from "axios";

import {
    DeleteFilled,
    DeleteOutlined,
    DislikeFilled,
    DislikeOutlined,
    LikeFilled,
    LikeOutlined
} from "@ant-design/icons";


const Polarizing = (props) => {

    const [BtnMouseOver, setBtnMouseOver] = useState(false);

    const [Like, setLike] = useState(false);
    const [Dislike, setDislike] = useState(false);

    useEffect(() => {

        const state = checkPolarized(props.id);

        if (state === "Like") {
            setLikeBtn();
        } else if (state === "Dislike") {
            setDislikeBtn();
        }
    }, []);

    const checkPolarized = (id) => {

        if (props.like_list && props.like_list.includes(id)) {
            return "Like";
        } else if (props.dislike_list && props.dislike_list.includes(id)) {
            return "Dislike";
        }
        return null;
    };

    const checkPolarizing = (id, like, dislike) => {

        // Like 또는 Dislike 클릭하여 선택 시: false
        // 이미 선택된 버튼을 다시 클릭하여 취소 시: true

        props.addList(id, (like === 'false') ? 'like' : ((dislike === 'false') ? 'dislike' : null));
    };

    const setLikeBtn = () => {

        if (Dislike) {
            setDislike(!Dislike);
        }
        setLike(!Like);

        if (props.is_search) {
            onSearchPageFunc(1);
        }
    }

    const setDislikeBtn = () => {

        if (Like) {
            setLike(!Like);
        }
        setDislike(!Dislike);

        if (props.is_search) {
            onSearchPageFunc(0);
        }
    }

    const onClickLikes = (e) => {

        setLikeBtn();
        checkPolarizing(props.id, e.currentTarget.value, null);
    };

    const onClickDislikes = (e) => {

        setDislikeBtn();
        checkPolarizing(props.id, null, e.currentTarget.value);
    };

    const onMouseOver = () => {

        setBtnMouseOver(!BtnMouseOver);
    };

    const onRemoveSong = () => {

        const song_id = {song_id: parseInt(props.id)};

        axios.post('http://127.0.0.1:8000/users/', song_id)
            .then(response => {
                if (response.data) {
                    window.location.replace("/admin/user-profile");
                } else {
                    alert('노래 삭제에 실패했습니다.');
                }
            })
            .catch(error => {
                console.log(error);
            })
    }

    const onSearchPageFunc = (state) => {

        const song_data = {
            song_id: parseInt(props.id),
            isLike: state
        };

        axios.post('http://127.0.0.1:8000/users/select_song', song_data)
            .then(response => {
                if (response.data) {
                    window.location.replace("/admin/search");
                } else {
                    alert('노래 추가에 실패했습니다.');
                }
            })
            .catch(error => {
                console.log(error);
            })
    };

    return (
        <>
            {
                props.btn_type === "delete" ?
                    (
                        <button className={BtnMouseOver ? "btn btn-icon btn-default" : "btn btn-icon btn-secondary"}
                                type="button" onMouseEnter={onMouseOver} onMouseLeave={onMouseOver}
                                onClick={onRemoveSong}>
                                <span key="button-basic-delete">
                                    {
                                        BtnMouseOver ? (<DeleteFilled/>) : (<DeleteOutlined/>)
                                    }
                                </span>
                        </button>
                    ) :
                    (
                        <>
                            <button className={Like ? "btn btn-icon btn-danger" : "btn btn-icon btn-secondary"}
                                    type="button" value={Like} onClick={onClickLikes}>
                                <span key="button-basic-like">
                                    {
                                        Like ? (<LikeFilled/>) : (<LikeOutlined/>)
                                    }
                                </span>
                            </button>
                            <button className={Dislike ? "btn btn-icon btn-default" : "btn btn-icon btn-secondary"}
                                    type="button" value={Dislike} onClick={onClickDislikes}>
                                <span key="button-basic-dislike">
                                    {
                                        Dislike ? (<DislikeFilled/>) : (<DislikeOutlined/>)
                                    }
                                </span>
                            </button>
                        </>
                    )
            }
        </>
    );
}

export default Polarizing;
